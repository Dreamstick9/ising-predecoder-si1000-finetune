# C++ pipeline benchmark

`src/qec_bench.cpp` times two decoding paths on the same recorded shots:

1. PyMatching (C++ core) on the raw detection events.
2. Pre-decoder (TensorRT engine) followed by PyMatching on the residual
   detection events, with the two logical predictions XOR-ed.

Timing is wall-clock and single-threaded: one shot at a time for matching, one
batch at a time for the GPU stage. The GPU stage includes host-device copies.

## Dependencies

- TensorRT 11.4 and the CUDA 13 runtime (`../environment/cpp_venv_pip_freeze.txt`)
- [PyMatching](https://github.com/oscarhiggott/PyMatching) built as a static
  library, with the [Stim](https://github.com/quantumlib/Stim) revision it
  pins. Both must be compiled with the same SIMD flags as the benchmark
  (`build.sh` uses `-mno-avx2 -msse2`).

`build.sh` records the include and library paths used. Adjust them to the
local installation.

## Steps

1. **Export the pre-decoder to ONNX** with NVIDIA's inference workflow (X
   basis). The file is written to the Ising-Decoding root as
   `predecoder_memory_d<d>_T<rounds>_X.onnx`.
   ```bash
   ONNX_WORKFLOW=1 ../training/evaluate.sh config_acc_si1000_p003 export_large_after st <checkpoint.safetensors> 4096 13 13
   ```
2. **Generate reference data** (detection events, observables, PyMatching
   predictions, detector error model). Run from the Ising-Decoding root:
   ```bash
   CUDA_VISIBLE_DEVICES="" PYTHONPATH=code python /path/to/cpp/src/make_cpp_data.py \
       --noise si1000 --p 0.003 --distance 13 --n-rounds 13 --basis X \
       --num-samples 50000 --output-dir data/d13_T13_X_si1000_p003
   ```
3. **Build a TensorRT engine** for a fixed batch size:
   ```bash
   python src/build_engine.py --onnx large_after_d13_T13_X.onnx --engine large_after_d13_T13_X_b64.engine --batch 64
   ```
4. **Build and run the benchmark:**
   ```bash
   ./build.sh
   build/qec_bench --data data/d13_T13_X_si1000_p003 --engine large_after_d13_T13_X_b64.engine
   ```

## Notes

- Engines are FP32. The exported network is strongly typed, so half precision
  requires an FP16 ONNX file; this was not done.
- Engines are specific to the TensorRT version and GPU model.
- X basis only.
- On one reference set, the TensorRT output differed from an ONNX Runtime CPU
  reference on 45 of 20,000 shots. The cause was not established.

The benchmark logs are in `../logs/bench_*.log`.
