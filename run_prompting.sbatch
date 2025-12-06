#!/bin/bash
#SBATCH --job-name=prompting
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=output_prompting_%j.out
#SBATCH --error=output_prompting_%j.err
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/ngc_pt_jan.toml

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

pip install accelerate
pip uninstall -y transformers tokenizers
pip install "tokenizers==0.20.0" "transformers==4.46.3"

mkdir -p /capstor/scratch/cscs/$USER/huggingface/
export HF_HOME=/capstor/scratch/cscs/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

export REPO_ROOT=/iopsstor/scratch/cscs/$USER/Romanian_Math_LLM

python3 /iopsstor/scratch/cscs/$USER/Romanian_Math_LLM/prompting_local.py
