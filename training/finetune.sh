#!/bin/bash
# usage: finetune.sh <experiment> <config> <init_weights.pt> <epochs> <samples_per_epoch> <lr> [ler_test_samples]
# Fine-tune a pre-decoder from existing weights with a fresh optimizer. The initial weights are placed
# in the run's models folder as the "epoch 0" file, which NVIDIA's resume logic then loads.
# Re-running the same command resumes from the last saved epoch.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; cd $REPO
EXP=$1; CFG=$2; INIT=$3; EPOCHS=$4; SAMPLES=$5; LR=$6; TEST=${7:-65536}
OUT=$P/runs/outputs/$EXP/models
mkdir -p $OUT
if ! ls $OUT/checkpoint.0.*.pt >/dev/null 2>&1 && [ ! -f $OUT/PreDecoderModelMemory_v1.0.0.pt ]; then
  cp "$INIT" $OUT/PreDecoderModelMemory_v1.0.0.pt
fi
export CONFIG_NAME=$CFG EXPERIMENT_NAME=$EXP WORKFLOW=train GPUS=1
export PREDECODER_TRAIN_EPOCHS=$EPOCHS PREDECODER_TRAIN_SAMPLES=$SAMPLES PREDECODER_FT_LR=$LR
export PREDECODER_VAL_SAMPLES=65536 PREDECODER_TEST_SAMPLES=$TEST PREDECODER_DISABLE_SDR=1
export PREDECODER_BASE_OUTPUT_DIR=$P/runs/outputs PREDECODER_LOG_BASE_DIR=$P/runs/logs PREDECODER_PYTHON=$P/venv/bin/python
bash code/scripts/local_run.sh
