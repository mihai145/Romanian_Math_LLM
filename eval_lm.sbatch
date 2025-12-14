#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=lm_eval_%j.out
#SBATCH --error=lm_eval_%j.err
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/my_env.toml

# sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_gsm8k" "chat"
# sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_mathqa" "chat"
# sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_gsm8k" "chat"
# sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_mathqa" "chat"

# sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_gsm8k" "nochat"
# sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_mathqa" "nochat"
# sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_gsm8k" "nochat"
# sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_mathqa" "nochat"

set -eo pipefail

# TODO: PUT YOUR HF_TOKEN here!!!
export HF_TOKEN=HF_TOKEN

MODEL="${1:?Missing MODEL arg (HF repo id or local path)}"
TASK="${2:?Missing TASK arg (e.g. ro_gsm8k_ro or ro_mathqa_ro_mc)}"
CHAT="${3:-chat}" # chat | nochat

CHAT_FLAG="--apply_chat_template"
if [[ "$CHAT" == "nochat" ]]; then
  CHAT_FLAG=""
fi

pip install -U \
  git+https://github.com/EleutherAI/lm-evaluation-harness.git \
  git+https://github.com/huggingface/transformers.git \
  accelerate datasets

mkdir -p /capstor/scratch/cscs/$USER/huggingface/
export HF_HOME=/capstor/scratch/cscs/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

export REPO_ROOT=/iopsstor/scratch/cscs/$USER/Romanian_Math_LLM

# filesystem-friendly model name
MODEL_TAG="$(echo "$MODEL" | tr '/:' '__')"
OUT="$REPO_ROOT/lm_eval_out/${MODEL_TAG}/${TASK}/${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "================ LM EVAL ================"
echo "SLURM_JOB_ID : $SLURM_JOB_ID"
echo "MODEL        : $MODEL"
echo "TASK         : $TASK"
echo "OUT          : $OUT"
echo "PWD          : $(pwd)"
echo "CHAT_FLAG    : $CHAT_FLAG"
echo "========================================="

export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

export UCX_TLS=sm,self
export UCX_MEMTYPE_CACHE=n

accelerate launch --num_processes 4 -m lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$TASK" \
  --include_path "$REPO_ROOT/lm_eval_tasks" \
  --batch_size auto \
  $CHAT_FLAG \
  --log_samples \
  --output_path "$OUT" \
  --write_out
