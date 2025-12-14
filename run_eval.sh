# Before finetuning

sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_gsm8k" "chat"
sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_mathqa" "chat"
sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_gsm8k" "chat"
sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_mathqa" "chat"

sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_gsm8k" "nochat"
sbatch eval_lm.sbatch "swiss-ai/Apertus-8B-Instruct-2509" "ro_mathqa" "nochat"
sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_gsm8k" "nochat"
sbatch eval_lm.sbatch "meta-llama/Llama-3.1-8B-Instruct" "ro_mathqa" "nochat"

# After finetuning

BASE=/capstor/scratch/cscs/moprea/finetuned
LLAMA=meta-llama__Llama-3.1-8B-Instruct
APERTUS=swiss-ai__Apertus-8B-Instruct-2509

MODELS=(
    $BASE/220/$LLAMA/ro_gsm8k/merged_bf16
    $BASE/220/$APERTUS/ro_gsm8k/merged_bf16
    $BASE/330/$LLAMA/ro_gsm8k/merged_bf16
    $BASE/330/$APERTUS/ro_gsm8k/merged_bf16
    $BASE/422/$LLAMA/ro_mathqa/merged_bf16
    $BASE/422/$APERTUS/ro_mathqa/merged_bf16
    $BASE/844/$LLAMA/ro_mathqa/merged_bf16
    $BASE/844/$APERTUS/ro_mathqa/merged_bf16
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model"
    sbatch eval_lm.sbatch "$model" "ro_gsm8k" "chat"
    sbatch eval_lm.sbatch "$model" "ro_gsm8k" "nochat"
    sbatch eval_lm.sbatch "$model" "ro_mathqa" "chat"
    sbatch eval_lm.sbatch "$model" "ro_mathqa" "nochat"
done
