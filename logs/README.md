# Raw logs

Unedited output of NVIDIA's Ising-Decoding workflows and of the C++ benchmark,
apart from one substitution: the host and account names were replaced with
`gpu-host` and `user`.

`analysis/make_figures.py` reads the `[Inference Summary]` line of each
measurement log, the per-epoch test lines of the training logs, and the
`bench_*` logs.

## Logical-error-rate measurements

Each was produced by `training/evaluate.sh` (or `evaluate_d9.sh` for the
`before_*` / `after1_*` logs) with the arguments shown. `config_public` is the
configuration shipped with Ising-Decoding (model 1, depolarizing noise,
p = 0.003); the others are in `training/configs/`. Shots are per basis; X and Z
are both measured.

| Log | Checkpoint | Noise | Config | d | Rounds | Shots per basis |
|---|---|---|---|---|---|---|
| `before_dep_p003` | Fast, published | Depolarizing, p=0.003 | `config_public` | 9 | 9 | 1,048,576 |
| `after1_dep_p003` | Fast, fine-tuned (run 1) | Depolarizing, p=0.003 | `config_public` | 9 | 9 | 1,048,576 |
| `before_si1000_p002` | Fast, published | SI1000, p=0.002 | `config_si1000_p002` | 9 | 9 | 1,048,576 |
| `after1_si1000_p002` | Fast, fine-tuned (run 1) | SI1000, p=0.002 | `config_si1000_p002` | 9 | 9 | 1,048,576 |
| `before_si1000_p003` | Fast, published | SI1000, p=0.003 | `config_si1000_p003` | 9 | 9 | 1,048,576 |
| `after1_si1000_p003` | Fast, fine-tuned (run 1) | SI1000, p=0.003 | `config_si1000_p003` | 9 | 9 | 1,048,576 |
| `d13_before_si1000_p003` | Fast, published | SI1000, p=0.003 | `config_si1000_p003` | 13 | 13 | 524,288 |
| `d13_after1_si1000_p003` | Fast, fine-tuned (run 1) | SI1000, p=0.003 | `config_si1000_p003` | 13 | 13 | 524,288 |
| `acc13_before_si1000_p003` | Accurate, published | SI1000, p=0.003 | `config_acc_si1000_p003` | 13 | 13 | 524,288 |
| `acc13_after3_si1000_p003` | Accurate, fine-tuned (run 3) | SI1000, p=0.003 | `config_acc_si1000_p003` | 13 | 13 | 1,048,576 |
| `acc13_after3_si1000_p002` | Accurate, fine-tuned (run 3) | SI1000, p=0.002 | `config_acc_si1000_p002` | 13 | 13 | 1,048,576 |
| `acc13_after3_dep_p003` | Accurate, fine-tuned (run 3) | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 13 | 1,048,576 |
| `acc13_before_si1000_p002` | Accurate, published | SI1000, p=0.002 | `config_acc_si1000_p002` | 13 | 13 | 1,048,576 |
| `acc13_before_dep_p003` | Accurate, published | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 13 | 1,048,576 |
| `d13_before_si1000_p002` | Fast, published | SI1000, p=0.002 | `config_si1000_p002` | 13 | 13 | 1,048,576 |
| `d13_after1_si1000_p002` | Fast, fine-tuned (run 1) | SI1000, p=0.002 | `config_si1000_p002` | 13 | 13 | 1,048,576 |
| `d13_before_dep_p003` | Fast, published | Depolarizing, p=0.003 | `config_public` | 13 | 13 | 1,048,576 |
| `d13_after1_dep_p003` | Fast, fine-tuned (run 1) | Depolarizing, p=0.003 | `config_public` | 13 | 13 | 1,048,576 |
| `lat9_before_si1000_p003` | Fast, published | SI1000, p=0.003 | `config_si1000_p003` | 9 | 9 | 262,144 |
| `lat9_after1_si1000_p003` | Fast, fine-tuned (run 1) | SI1000, p=0.003 | `config_si1000_p003` | 9 | 9 | 262,144 |
| `lat9_before_si1000_p002` | Fast, published | SI1000, p=0.002 | `config_si1000_p002` | 9 | 9 | 262,144 |
| `lat9_after1_si1000_p002` | Fast, fine-tuned (run 1) | SI1000, p=0.002 | `config_si1000_p002` | 9 | 9 | 262,144 |
| `d17_before_si1000_p003` | Fast, published | SI1000, p=0.003 | `config_si1000_p003` | 17 | 17 | 262,144 |
| `d17_after1_si1000_p003` | Fast, fine-tuned (run 1) | SI1000, p=0.003 | `config_si1000_p003` | 17 | 17 | 262,144 |
| `r26_acc_before_dep_p003` | Accurate, published | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 26 | 262,144 |
| `r26_fast_before_dep_p003` | Fast, published | Depolarizing, p=0.003 | `config_public` | 13 | 26 | 262,144 |
| `r52_acc_before_dep_p003` | Accurate, published | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 52 | 262,144 |
| `r52_fast_before_dep_p003` | Fast, published | Depolarizing, p=0.003 | `config_public` | 13 | 52 | 262,144 |
| `r52_acc_before_si1000_p003` | Accurate, published | SI1000, p=0.003 | `config_acc_si1000_p003` | 13 | 52 | 131,072 |
| `r52_acc_after3_si1000_p003` | Accurate, fine-tuned (run 3) | SI1000, p=0.003 | `config_acc_si1000_p003` | 13 | 52 | 131,072 |
| `r52_fast_before_si1000_p003` | Fast, published | SI1000, p=0.003 | `config_si1000_p003` | 13 | 52 | 131,072 |
| `r52_fast_after1_si1000_p003` | Fast, fine-tuned (run 1) | SI1000, p=0.003 | `config_si1000_p003` | 13 | 52 | 131,072 |
| `r52_acc_before_si1000_p002` | Accurate, published | SI1000, p=0.002 | `config_acc_si1000_p002` | 13 | 52 | 262,144 |
| `r52_acc_after3_si1000_p002` | Accurate, fine-tuned (run 3) | SI1000, p=0.002 | `config_acc_si1000_p002` | 13 | 52 | 262,144 |
| `r52_fast_before_si1000_p002` | Fast, published | SI1000, p=0.002 | `config_si1000_p002` | 13 | 52 | 262,144 |
| `r52_fast_after1_si1000_p002` | Fast, fine-tuned (run 1) | SI1000, p=0.002 | `config_si1000_p002` | 13 | 52 | 262,144 |
| `r52_acc_after3_dep_p003` | Accurate, fine-tuned (run 3) | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 52 | 262,144 |
| `r52_fast_after1_dep_p003` | Fast, fine-tuned (run 1) | Depolarizing, p=0.003 | `config_public` | 13 | 52 | 262,144 |
| `d17r34_acc_before_si1000_p003` | Accurate, published | SI1000, p=0.003 | `config_acc_si1000_p003` | 17 | 34 | 131,072 |
| `d17r34_acc_after3_si1000_p003` | Accurate, fine-tuned (run 3) | SI1000, p=0.003 | `config_acc_si1000_p003` | 17 | 34 | 131,072 |
| `acc13_after4_si1000_p002` | Accurate, fine-tuned (run 4) | SI1000, p=0.002 | `config_acc_si1000_p002` | 13 | 13 | 1,048,576 |
| `acc13_after4_si1000_p003` | Accurate, fine-tuned (run 4) | SI1000, p=0.003 | `config_acc_si1000_p003` | 13 | 13 | 1,048,576 |
| `acc13_after4_dep_p003` | Accurate, fine-tuned (run 4) | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 13 | 1,048,576 |
| `r52_acc_after4_si1000_p002` | Accurate, fine-tuned (run 4) | SI1000, p=0.002 | `config_acc_si1000_p002` | 13 | 52 | 262,144 |
| `r52_acc_after4_si1000_p003` | Accurate, fine-tuned (run 4) | SI1000, p=0.003 | `config_acc_si1000_p003` | 13 | 52 | 131,072 |
| `r52_acc_after4_dep_p003` | Accurate, fine-tuned (run 4) | Depolarizing, p=0.003 | `config_acc_dep_p003` | 13 | 52 | 262,144 |

## Training, conversion, export and benchmark logs

| Log | Contents |
|---|---|
| `ft1_train` | Fine-tuning run 1 (Fast model, SI1000 p=0.003) |
| `ft3_train_b512` | Fine-tuning run 3 (Accurate model, SI1000 p=0.003) |
| `ft4_train_b512` | Fine-tuning run 4 (Accurate model, SI1000 p=0.002) |
| `ft1_convert` | Conversion of the run 1 checkpoint to fp16 safetensors |
| `ft3_convert` | Conversion of the run 3 checkpoint to fp16 safetensors |
| `ft4_convert` | Conversion of the run 4 checkpoint to fp16 safetensors |
| `export_small_before` | ONNX export, published Fast model, d=9, 9 rounds, X basis |
| `export_small_after` | ONNX export, run 1 model, d=9, 9 rounds, X basis |
| `export_large_before` | ONNX export, published Accurate model, d=13, 13 rounds, X basis |
| `export_large_after` | ONNX export, run 3 model, d=13, 13 rounds, X basis |
| `export_nvidia_d9` | ONNX export, published Fast model, d=9, 9 rounds (benchmark validation on depolarizing noise) |
| `make_cpp_data_d9` | Benchmark reference data, SI1000 p=0.003, d=9, 9 rounds, 50,000 shots |
| `make_cpp_data_d13` | Benchmark reference data, SI1000 p=0.003, d=13, 13 rounds, 50,000 shots |
| `make_cpp_data_dep` | Benchmark reference data, depolarizing noise (benchmark validation) |
| `bench_large_after_b1` | C++ benchmark, run 3 (Accurate), d=13, batch size 1 |
| `bench_large_after_b1024` | C++ benchmark, run 3 (Accurate), d=13, batch size 1024 |
| `bench_large_after_b1024_fp16` | C++ benchmark, run 3 (Accurate), d=13, batch size 1024, engine built with `--fp16` (the flag has no effect on this strongly typed network, so the engine is FP32) |
| `bench_large_after_b1_fp16` | C++ benchmark, run 3 (Accurate), d=13, batch size 1, engine built with `--fp16` (the flag has no effect on this strongly typed network, so the engine is FP32) |
| `bench_large_after_b64` | C++ benchmark, run 3 (Accurate), d=13, batch size 64 |
| `bench_large_before_b1` | C++ benchmark, published Accurate, d=13, batch size 1 |
| `bench_large_before_b64` | C++ benchmark, published Accurate, d=13, batch size 64 |
| `bench_small_after_b1` | C++ benchmark, run 1 (Fast), d=9, batch size 1 |
| `bench_small_after_b64` | C++ benchmark, run 1 (Fast), d=9, batch size 64 |
| `bench_small_before_b1` | C++ benchmark, published Fast, d=9, batch size 1 |
| `bench_small_before_b64` | C++ benchmark, published Fast, d=9, batch size 64 |
