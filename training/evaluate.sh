#!/bin/bash
# usage: evaluate.sh <config_name> <experiment_name> <pt|st> <checkpoint_path> <shots_per_basis> <distance> <rounds>
# Logical error rate at any distance and round count: PyMatching alone vs pre-decoder + PyMatching.
# Keeps NVIDIA's single-shot PyMatching latency measurement on (10,000 CPU decodes).
# <pt|st> selects a PyTorch checkpoint or a safetensors file.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; cd $REPO
CFG=$1; EXP=$2; KIND=$3; CK=$4; SHOTS=$5; DIST=$6; ROUNDS=$7
export CONFIG_NAME=$CFG EXPERIMENT_NAME=$EXP WORKFLOW=inference GPUS=1 DISTANCE=$DIST N_ROUNDS=$ROUNDS
export PREDECODER_BASE_OUTPUT_DIR=$P/runs/outputs PREDECODER_LOG_BASE_DIR=$P/runs/logs PREDECODER_PYTHON=$P/venv/bin/python
export PREDECODER_EMIT_INFERENCE_SUMMARY=1 PREDECODER_INFERENCE_NUM_SAMPLES=$SHOTS
if [ "$KIND" = st ]; then export PREDECODER_SAFETENSORS_CHECKPOINT=$CK; else export PREDECODER_MODEL_CHECKPOINT_FILE=$CK; fi
bash code/scripts/local_run.sh
