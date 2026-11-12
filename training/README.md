# Fine-tuning and evaluation

Thin wrappers around the training and inference workflows of NVIDIA's
[Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding). All training and
evaluation logic is NVIDIA's; this folder adds configuration files, a small
patch, and launch scripts.

| File | Purpose |
|---|---|
| `configs/` | Noise-model and model-size configurations (see below) |
| `config_validator.patch` | Adds `PREDECODER_FT_LR` and `PREDECODER_FT_BATCH` overrides to `code/workflows/config_validator.py` |
| `env.sh` | Paths (`QEC_ROOT`, `ISING_DECODING_DIR`) |
| `make_init_weights.py` | Converts a published fp16 safetensors checkpoint to fp32 initial weights |
| `finetune.sh` | Launches a fine-tuning run |
| `evaluate.sh` | Logical-error-rate measurement at a given distance and round count |
| `evaluate_d9.sh` | Same at distance 9, 9 rounds, without the latency measurement |
| `summarize.py` | Tabulates `[Inference Summary]` lines with standard errors |

## Configurations

| Config | Model | Noise |
|---|---|---|
| `config_si1000_p003` | 1 (Fast, R = 9) | SI1000, p = 0.003 |
| `config_si1000_p002` | 1 (Fast, R = 9) | SI1000, p = 0.002 |
| `config_acc_si1000_p003` | 4 (Accurate, R = 13) | SI1000, p = 0.003 |
| `config_acc_si1000_p002` | 4 (Accurate, R = 13) | SI1000, p = 0.002 |
| `config_acc_dep_p003` | 4 (Accurate, R = 13) | Circuit-level depolarizing, p = 0.003 |

The SI1000 parameters are the 25-parameter circuit-level noise specification
produced by `NoiseModel.from_si1000(p)` in Ising-Decoding. Depolarizing noise
for model 1 uses the `conf/config_public.yaml` shipped with Ising-Decoding.

## Setup

1. Clone Ising-Decoding and install its requirements into a Python 3.12
   environment. The package versions used are listed in
   `../environment/venv_pip_freeze.txt`.
2. Apply the patch and copy the configurations:
   ```bash
   cd "$ISING_DECODING_DIR"
   patch -p1 < /path/to/training/config_validator.patch
   cp /path/to/training/configs/*.yaml conf/
   ```
3. Download the published fp16 safetensors checkpoints from NVIDIA's Hugging
   Face pages
   ([Fast](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast),
   [Accurate](https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Accurate))
   after accepting their terms.
4. Set `QEC_ROOT` (working directory containing `venv/`) and
   `ISING_DECODING_DIR`, or edit `env.sh`.
5. Convert each published checkpoint to fp32 initial weights with
   `make_init_weights.py` (usage in its header).

## Fine-tuning runs

```
finetune.sh <experiment> <config> <init_weights.pt> <epochs> <samples_per_epoch> <lr> [ler_test_samples]
```

| Run | Command | Wall-clock time | Log |
|---|---|---|---|
| 1 | `finetune.sh ft1_si1000 config_si1000_p003 nvidia_fast_init_fp32.pt 14 8388608 1e-4 131072` | 7 h 20 min | `ft1_train` |
| 3 | `PREDECODER_FT_BATCH=512 finetune.sh ft3_acc_si1000 config_acc_si1000_p003 nvidia_accurate_init_fp32.pt 16 1048576 7e-5 131072` | 6 h 36 min | `ft3_train_b512` |
| 4 | `PREDECODER_FT_BATCH=512 finetune.sh ft4_acc_si1000_p002 config_acc_si1000_p002 nvidia_accurate_init_fp32.pt 16 1048576 7e-5 262144` | 6 h 28 min | `ft4_train_b512` |

Run 1 uses NVIDIA's default batch schedule (512 in the first epoch, 2048
afterwards). Runs 3 and 4 use a fixed batch size of 512. The optimizer is
NVIDIA's default (Lion) with a fresh state. Run 2 was not carried out.

After training, the last epoch's checkpoint was converted with NVIDIA's
`code/export/checkpoint_to_safetensors.py --fp16` (`--model-id 1` or `4`).

## Evaluation

```
evaluate.sh <config> <experiment> <pt|st> <checkpoint> <shots_per_basis> <distance> <rounds>
```

The exact arguments of every reported measurement are tabulated in
[`../logs/README.md`](../logs/README.md). The improvement factor and its
standard error are computed from the `[Inference Summary]` line by
`summarize.py` and by `../analysis/make_figures.py`.

Sampling is not seeded, so repeated measurements agree statistically, not
bit for bit.
