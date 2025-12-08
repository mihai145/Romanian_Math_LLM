#!/bin/bash
#SBATCH --job-name=eval_llms
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=output_eval_%j.out
#SBATCH --error=output_eval_%j.err
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/ngc_pt_jan.toml

set -eo pipefail

pip install accelerate
pip uninstall -y transformers tokenizers
pip install "tokenizers==0.20.0" "transformers==4.46.3"

mkdir -p /capstor/scratch/cscs/$USER/huggingface/
export HF_HOME=/capstor/scratch/cscs/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

export REPO_ROOT=/iopsstor/scratch/cscs/$USER/project/Romanian_Math_LLM
export OPENAI_API_KEY=[YOUR_KEY_HERE]

python3 $REPO_ROOT/evaluate_models_local.py
