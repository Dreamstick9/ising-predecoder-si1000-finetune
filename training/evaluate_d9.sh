#!/bin/bash
# usage: evaluate_d9.sh <config_name> <experiment_name> <pt|st> <checkpoint_path> [shots_per_basis]
# Logical error rate at distance 9, 9 rounds: PyMatching alone vs pre-decoder + PyMatching.
# The single-shot matching latency measurement is switched off.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; cd $REPO
CFG=$1; EXP=$2; KIND=$3; CK=$4; SHOTS=${5:-262144}
export CONFIG_NAME=$CFG EXPERIMENT_NAME=$EXP WORKFLOW=inference GPUS=1 DISTANCE=9 N_ROUNDS=9
export PREDECODER_BASE_OUTPUT_DIR=$P/runs/outputs PREDECODER_LOG_BASE_DIR=$P/runs/logs PREDECODER_PYTHON=$P/venv/bin/python
export PREDECODER_EMIT_INFERENCE_SUMMARY=1 PREDECODER_INFERENCE_LATENCY_SAMPLES=0 PREDECODER_INFERENCE_NUM_SAMPLES=$SHOTS
if [ "$KIND" = st ]; then export PREDECODER_SAFETENSORS_CHECKPOINT=$CK; else export PREDECODER_MODEL_CHECKPOINT_FILE=$CK; fi
bash code/scripts/local_run.sh
