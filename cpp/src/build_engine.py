#!/usr/bin/env python
"""Build a TensorRT engine from an exported pre-decoder ONNX file.

The exported graph has a dynamic batch dimension. The engine is built for one
fixed batch size, which is what the C++ benchmark (qec_bench) reads back.

    python build_engine.py --onnx predecoder.onnx --engine predecoder_b1.engine --batch 1
"""
import argparse
import sys
import time

import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", help="allow FP16 layers (ignored if this TensorRT build has no such flag)")
    a = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(a.onnx):
        for i in range(parser.num_errors):
            print("parse error:", parser.get_error(i), file=sys.stderr)
        sys.exit(1)

    inp = network.get_input(0)
    out = network.get_output(0)
    fixed = (a.batch,) + tuple(inp.shape)[1:]
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, fixed, fixed, fixed)
    config.add_optimization_profile(profile)

    precision = "fp32"
    if a.fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
            precision = "fp16 allowed"
        except Exception as e:  # report, do not hide
            print(f"FP16 flag not applied: {type(e).__name__}: {e}", file=sys.stderr)

    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        print("engine build failed", file=sys.stderr)
        sys.exit(1)
    with open(a.engine, "wb") as f:
        f.write(blob)
    print(
        f"TensorRT {trt.__version__} | input {inp.name} {fixed} {inp.dtype} | "
        f"output {out.name} {tuple(out.shape)} {out.dtype} | precision {precision} | "
        f"built in {time.time() - t0:.1f}s -> {a.engine}"
    )


if __name__ == "__main__":
    main()
