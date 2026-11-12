#!/bin/bash
# Compile qec_bench against the locally built PyMatching/Stim libraries, TensorRT, and the CUDA runtime.
source "$(dirname "${BASH_SOURCE[0]}")/../training/env.sh"
C=$P/cpp; D=$C/deps
STIM=$D/Stim-1320ad7eac7de34d2e9c70daa44fbc6d84174450
B=$C/build/pymatching
TRT_LIB=$C/venv/lib/python3.12/site-packages/tensorrt_libs
CUDA=$P/venv/lib/python3.12/site-packages/nvidia/cu13
# -mno-avx2 -msse2 must match how libstim/libpymatching were built (Stim's SIMD word size).
g++ -O3 -std=gnu++20 -mno-avx2 -msse2 -include cstdint \
  -I$D/PyMatching-master/src -I$STIM/src -I$D/TensorRT-11.4/include -I$CUDA/include -I$P/venv/lib/python3.12/site-packages/triton/backends/nvidia/include \
  $C/src/qec_bench.cpp $B/libpymatching.a $B/_deps/stim-build/out/libstim.a \
  $TRT_LIB/libnvinfer.so.11 $CUDA/lib/libcudart.so.13 \
  -Wl,-rpath,$TRT_LIB -Wl,-rpath,$CUDA/lib -lpthread -ldl -o $C/build/qec_bench
