#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=sft_%j.out
#SBATCH --error=sft_%j.err
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/my_env.toml

# sbatch run_sft.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_gsm8k"
# sbatch run_sft.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_mathqa"
# sbatch run_sft.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_gsm8k"
# sbatch run_sft.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_mathqa"

set -eo pipefail

# TODO: PUT YOUR HF_TOKEN here!!!
export HF_TOKEN=HF_TOKEN

MODEL="${1:?Missing MODEL}"
DATASET="${2:?Missing DATASET}"

export REPO_ROOT=/iopsstor/scratch/cscs/$USER/Romanian_Math_LLM

mkdir -p /capstor/scratch/cscs/$USER/huggingface/
export HF_HOME=/capstor/scratch/cscs/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

export MAX_SEQ_LEN=512
export BATCH=1
export GRAD_ACCUM=16
export LR=1e-4

# # Do 2 epochs on the smaller dataset, 1 epoch on the bigger dataset
# case "$DATASET" in
#   ro_gsm8k)  export MAX_STEPS=$((2 * 110))  ;;  # 7k / 64 ≈ 110 steps/epoch
#   ro_mathqa) export MAX_STEPS=$((1 * 422))  ;;  # 27k / 64 ≈ 422 steps/epoch
# esac

# Do 3 epochs on the smaller dataset, 2 epoch on the bigger dataset
case "$DATASET" in
  ro_gsm8k)  export MAX_STEPS=$((3 * 110))  ;;  # 7k / 64 ≈ 110 steps/epoch
  ro_mathqa) export MAX_STEPS=$((2 * 422))  ;;  # 27k / 64 ≈ 422 steps/epoch
esac

pip install -U \
  "accelerate>=0.34.0" \
  "datasets>=2.20.0" \
  "transformers>=4.45.0" \
  "trl>=0.11.4" \
  "peft>=0.12.0" \
  "bitsandbytes>=0.43.0" \
  "safetensors" \
  "sentencepiece" \
  "evaluate"

pip uninstall -y apex || true

export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export UCX_TLS=sm,self
export UCX_MEMTYPE_CACHE=n

echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "MAX_STEPS=$MAX_STEPS MAX_SEQ_LEN=$MAX_SEQ_LEN"

accelerate launch --num_processes 4 \
  "$REPO_ROOT/train_sft.py" \
  "$MODEL" \
  "$DATASET"
