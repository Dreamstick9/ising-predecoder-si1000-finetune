// qec_bench: C++ benchmark for a pre-decoder + PyMatching decoding pipeline.
//
// Two paths are measured on the same recorded shots:
//   1. Standard decoder alone: PyMatching (C++ core) on the raw detector bits.
//   2. Pre-decoder + PyMatching: a TensorRT engine turns detector bits into
//      [logical flip, residual detector bits]; PyMatching decodes the residual
//      and its answer is XOR-ed with the logical flip.
//
// Inputs come from a data directory written by make_cpp_data.py:
//   detectors.bin, observables.bin, pymatching_predictions.bin  (uint32 rows, uint32 cols, int32 body)
//   predecoder_outputs.bin                                      (uint32 rows, uint32 cols, uint8 body; optional)
//   model.dem                                                   (Stim detector error model, text)
//   metadata.txt                                                (key=value lines, including n_rounds)
//
// Usage:
//   qec_bench --data DIR [--engine FILE] [--warmup N] [--limit N]
//
// Timing is wall-clock, single thread, one shot at a time for the matching
// stage and one batch at a time for the GPU stage (batch size is fixed in the
// engine). "Per round" divides by n_rounds from metadata.txt.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"

namespace {

using Clock = std::chrono::steady_clock;

double micros(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

std::vector<char> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail("cannot open " + path);
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(n));
    if (n > 0 && !f.read(buf.data(), n)) fail("cannot read " + path);
    return buf;
}

// A row-major matrix of bytes (values 0/1), whatever the on-disk element type.
struct ByteMatrix {
    uint32_t rows = 0, cols = 0;
    std::vector<uint8_t> data;
    const uint8_t* row(size_t r) const { return data.data() + r * cols; }
};

ByteMatrix read_matrix(const std::string& path, size_t elem_size) {
    std::vector<char> raw = read_bytes(path);
    if (raw.size() < 8) fail("file too small: " + path);
    ByteMatrix m;
    std::memcpy(&m.rows, raw.data(), 4);
    std::memcpy(&m.cols, raw.data() + 4, 4);
    size_t n = static_cast<size_t>(m.rows) * m.cols;
    if (raw.size() != 8 + n * elem_size) {
        fail("unexpected size for " + path + ": header says " + std::to_string(m.rows) + "x" +
             std::to_string(m.cols) + " with " + std::to_string(elem_size) + "-byte elements");
    }
    m.data.resize(n);
    if (elem_size == 1) {
        std::memcpy(m.data.data(), raw.data() + 8, n);
    } else if (elem_size == 4) {
        const char* p = raw.data() + 8;
        for (size_t i = 0; i < n; i++) {
            int32_t v;
            std::memcpy(&v, p + 4 * i, 4);
            m.data[i] = static_cast<uint8_t>(v != 0);
        }
    } else {
        fail("unsupported element size");
    }
    return m;
}

std::map<std::string, std::string> read_metadata(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return kv;
}

struct Stats {
    double mean = 0, p50 = 0, p99 = 0, min = 0, max = 0;
    size_t n = 0;
};

Stats summarize(std::vector<double> v) {
    Stats s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    s.p50 = v[v.size() / 2];
    s.p99 = v[std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.99))];
    s.min = v.front();
    s.max = v.back();
    return s;
}

// Decode one shot's detector bits with PyMatching. Returns the predicted
// logical observable bit (the pipeline has exactly one observable).
struct Matcher {
    pm::UserGraph graph;
    pm::Mwpm* mwpm = nullptr;
    size_t num_detectors = 0;
    std::vector<uint64_t> events;
    std::vector<uint8_t> obs;

    explicit Matcher(const std::string& dem_text) {
        stim::DetectorErrorModel dem(dem_text);
        graph = pm::detector_error_model_to_user_graph(dem, /*enable_correlations=*/false, pm::NUM_DISTINCT_WEIGHTS);
        mwpm = &graph.get_mwpm();
        num_detectors = graph.get_num_detectors();
        size_t num_obs = graph.get_num_observables();
        if (num_obs != 1) fail("expected exactly 1 observable, found " + std::to_string(num_obs));
        obs.assign(num_obs, 0);
        events.reserve(256);
    }

    uint8_t decode(const uint8_t* dets, size_t n, size_t* num_events = nullptr) {
        events.clear();
        for (size_t i = 0; i < n; i++) {
            if (dets[i]) events.push_back(i);
        }
        if (num_events) *num_events = events.size();
        obs[0] = 0;
        pm::total_weight_int weight = 0;
        pm::decode_detection_events(*mwpm, events, obs.data(), weight, /*edge_correlations=*/false);
        return obs[0] & 1;
    }
};

class TrtLogger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
};

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

// Runs the pre-decoder engine on fixed-size batches of detector bits.
struct PreDecoder {
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string in_name, out_name;
    size_t batch = 0, in_width = 0, out_width = 0;
    uint8_t* h_in = nullptr;
    uint8_t* h_out = nullptr;
    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaStream_t stream = nullptr;

    explicit PreDecoder(const std::string& engine_path) {
        std::vector<char> blob = read_bytes(engine_path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) fail("could not create TensorRT runtime");
        engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine) fail("could not load engine " + engine_path + " (engines are tied to the TensorRT version and GPU)");
        context.reset(engine->createExecutionContext());
        if (!context) fail("could not create execution context");

        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                in_name = name;
            } else {
                out_name = name;
            }
        }
        if (in_name.empty() || out_name.empty()) fail("engine must have one input and one output");
        if (engine->getTensorDataType(in_name.c_str()) != nvinfer1::DataType::kUINT8 ||
            engine->getTensorDataType(out_name.c_str()) != nvinfer1::DataType::kUINT8) {
            fail("engine input and output must both be uint8");
        }

        // The batch dimension is dynamic in the exported graph; the engine was
        // built with a single fixed batch size, read here from its profile.
        nvinfer1::Dims in_dims =
            engine->getProfileShape(in_name.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
        if (in_dims.nbDims != 2) fail("expected a 2-D input (batch, detectors)");
        batch = static_cast<size_t>(in_dims.d[0]);
        in_width = static_cast<size_t>(in_dims.d[1]);
        if (!context->setInputShape(in_name.c_str(), in_dims)) fail("setInputShape failed");
        nvinfer1::Dims out_dims = context->getTensorShape(out_name.c_str());
        if (out_dims.nbDims != 2 || static_cast<size_t>(out_dims.d[0]) != batch) fail("unexpected output shape");
        out_width = static_cast<size_t>(out_dims.d[1]);

        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
        void* pinned_in = nullptr;
        void* pinned_out = nullptr;
        cuda_check(cudaMallocHost(&pinned_in, batch * in_width), "cudaMallocHost(in)");
        cuda_check(cudaMallocHost(&pinned_out, batch * out_width), "cudaMallocHost(out)");
        h_in = static_cast<uint8_t*>(pinned_in);
        h_out = static_cast<uint8_t*>(pinned_out);
        cuda_check(cudaMalloc(&d_in, batch * in_width), "cudaMalloc(in)");
        cuda_check(cudaMalloc(&d_out, batch * out_width), "cudaMalloc(out)");
        if (!context->setTensorAddress(in_name.c_str(), d_in)) fail("setTensorAddress(in) failed");
        if (!context->setTensorAddress(out_name.c_str(), d_out)) fail("setTensorAddress(out) failed");
    }

    ~PreDecoder() {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (h_in) cudaFreeHost(h_in);
        if (h_out) cudaFreeHost(h_out);
        if (stream) cudaStreamDestroy(stream);
    }

    // Host input buffer -> GPU -> engine -> host output buffer. Blocking.
    void run() {
        cuda_check(cudaMemcpyAsync(d_in, h_in, batch * in_width, cudaMemcpyHostToDevice, stream), "copy in");
        if (!context->enqueueV3(stream)) fail("enqueueV3 failed");
        cuda_check(cudaMemcpyAsync(h_out, d_out, batch * out_width, cudaMemcpyDeviceToHost, stream), "copy out");
        cuda_check(cudaStreamSynchronize(stream), "synchronize");
    }
};

void print_stats(const char* label, const Stats& s, double rounds) {
    std::printf("  %-34s mean %9.2f  p50 %9.2f  p99 %9.2f  min %9.2f  us   | per round: mean %7.3f  p50 %7.3f us\n",
                label, s.mean, s.p50, s.p99, s.min, s.mean / rounds, s.p50 / rounds);
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir, engine_path;
    size_t warmup = 200, limit = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail("missing value for " + a);
            return argv[++i];
        };
        if (a == "--data") data_dir = next();
        else if (a == "--engine") engine_path = next();
        else if (a == "--warmup") warmup = std::stoul(next());
        else if (a == "--limit") limit = std::stoul(next());
        else fail("unknown argument " + a);
    }
    if (data_dir.empty()) fail("usage: qec_bench --data DIR [--engine FILE] [--warmup N] [--limit N]");

    auto meta = read_metadata(data_dir + "/metadata.txt");
    double rounds = meta.count("n_rounds") ? std::stod(meta["n_rounds"]) : 1.0;

    ByteMatrix dets = read_matrix(data_dir + "/detectors.bin", 4);
    ByteMatrix obs = read_matrix(data_dir + "/observables.bin", 4);
    ByteMatrix py_pred = read_matrix(data_dir + "/pymatching_predictions.bin", 4);
    if (obs.rows != dets.rows || py_pred.rows != dets.rows) fail("row counts differ between data files");
    if (obs.cols != 1) fail("expected one observable column");
    size_t shots = dets.rows;
    if (limit > 0 && limit < shots) shots = limit;

    std::vector<char> dem_raw = read_bytes(data_dir + "/model.dem");
    Matcher matcher(std::string(dem_raw.begin(), dem_raw.end()));
    if (matcher.num_detectors > dets.cols) fail("model.dem has more detectors than the data");

    std::printf("data: %s\n", data_dir.c_str());
    std::printf("shots %zu | detectors %u | rounds %.0f | distance %s | noise %s p=%s\n", shots, dets.cols, rounds,
                meta.count("distance") ? meta["distance"].c_str() : "?",
                meta.count("noise") ? meta["noise"].c_str() : "?",
                meta.count("noise_p") ? meta["noise_p"].c_str() : (meta.count("p_error") ? meta["p_error"].c_str() : "?"));

    // ---- Path 1: standard decoder alone ----------------------------------
    size_t base_errors = 0, base_mismatch_python = 0, base_events = 0;
    std::vector<double> base_times;
    base_times.reserve(shots);
    for (size_t s = 0; s < shots; s++) {
        size_t n_events = 0;
        auto t0 = Clock::now();
        uint8_t pred = matcher.decode(dets.row(s), dets.cols, &n_events);
        auto t1 = Clock::now();
        if (s >= warmup) base_times.push_back(micros(t0, t1));
        base_events += n_events;
        base_errors += (pred != obs.row(s)[0]);
        base_mismatch_python += (pred != py_pred.row(s)[0]);
    }
    Stats base = summarize(base_times);

    std::printf("\n[1] standard decoder alone (PyMatching, C++)\n");
    std::printf("  logical errors %zu / %zu  (rate %.3e)\n", base_errors, shots, double(base_errors) / shots);
    std::printf("  disagreements with Python PyMatching: %zu\n", base_mismatch_python);
    std::printf("  mean detection events per shot: %.2f\n", double(base_events) / shots);
    print_stats("decode time per shot", base, rounds);

    if (engine_path.empty()) {
        std::printf("\n(no --engine given; pre-decoder path skipped)\n");
        return 0;
    }

    // ---- Path 2: pre-decoder + standard decoder ----------------------------
    PreDecoder pre(engine_path);
    if (pre.in_width != dets.cols) fail("engine input width does not match the data");
    if (pre.out_width != dets.cols + 1) fail("engine output width should be detectors + 1");

    ByteMatrix ref_out;
    bool have_ref = file_exists(data_dir + "/predecoder_outputs.bin");
    if (have_ref) {
        ref_out = read_matrix(data_dir + "/predecoder_outputs.bin", 1);
        if (ref_out.rows != dets.rows || ref_out.cols != pre.out_width) fail("predecoder_outputs.bin has the wrong shape");
    }

    size_t B = pre.batch;
    size_t pipe_errors = 0, resid_events = 0, ref_mismatch_bits = 0, ref_mismatch_shots = 0;
    std::vector<double> gpu_times, match_times, total_times;
    size_t batch_index = 0;
    for (size_t start = 0; start < shots; start += B, batch_index++) {
        size_t valid = std::min(B, shots - start);
        std::memset(pre.h_in, 0, B * pre.in_width);
        std::memcpy(pre.h_in, dets.row(start), valid * pre.in_width);

        auto t0 = Clock::now();
        pre.run();
        auto t1 = Clock::now();
        double gpu_us = micros(t0, t1);
        bool timed = batch_index * B >= warmup && valid == B;
        if (timed) gpu_times.push_back(gpu_us);

        for (size_t k = 0; k < valid; k++) {
            const uint8_t* out = pre.h_out + k * pre.out_width;
            size_t n_events = 0;
            auto t2 = Clock::now();
            uint8_t pre_logical = out[0] & 1;
            uint8_t pred = matcher.decode(out + 1, dets.cols, &n_events) ^ pre_logical;
            auto t3 = Clock::now();
            double match_us = micros(t2, t3);
            if (timed) {
                match_times.push_back(match_us);
                total_times.push_back(gpu_us / B + match_us);
            }
            resid_events += n_events;
            pipe_errors += (pred != obs.row(start + k)[0]);
            if (have_ref) {
                const uint8_t* ref = ref_out.row(start + k);
                size_t diff = 0;
                for (size_t j = 0; j < pre.out_width; j++) diff += (ref[j] != out[j]);
                ref_mismatch_bits += diff;
                ref_mismatch_shots += (diff != 0);
            }
        }
    }
    Stats gpu = summarize(gpu_times), match = summarize(match_times), total = summarize(total_times);

    std::printf("\n[2] pre-decoder (TensorRT, batch %zu) + PyMatching\n", B);
    std::printf("  engine: %s\n", engine_path.c_str());
    std::printf("  logical errors %zu / %zu  (rate %.3e)\n", pipe_errors, shots, double(pipe_errors) / shots);
    std::printf("  mean detection events per shot after pre-decoder: %.2f  (was %.2f)\n", double(resid_events) / shots,
                double(base_events) / shots);
    if (have_ref) {
        std::printf("  shots whose engine output differs from the Python reference: %zu / %zu  (%zu bits)\n",
                    ref_mismatch_shots, shots, ref_mismatch_bits);
    }
    print_stats("GPU stage per batch (incl. copies)", gpu, rounds);
    if (B > 1) {
        Stats per_shot = gpu;
        per_shot.mean /= B; per_shot.p50 /= B; per_shot.p99 /= B; per_shot.min /= B;
        print_stats("GPU stage per shot (batch / size)", per_shot, rounds);
    }
    print_stats("matching per shot (residual)", match, rounds);
    print_stats("total per shot (GPU/size + match)", total, rounds);

    std::printf("\nsummary: baseline %.3f us/round | pipeline %.3f us/round | ratio %.2fx | LER %.3e -> %.3e\n",
                base.mean / rounds, total.mean / rounds, base.mean / std::max(total.mean, 1e-12),
                double(base_errors) / shots, double(pipe_errors) / shots);
    return 0;
}
