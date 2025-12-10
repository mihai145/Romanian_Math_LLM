import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import os
import re
from tqdm import tqdm  # <-- NEW: Import tqdm

# Set environment variable (assuming REPO_ROOT is correctly set in your environment)
REPO_ROOT = os.getenv("REPO_ROOT", ".") # Defaulting to '.' if not set

# === MODIFIED SYSTEM PROMPT ===
system_prompt = """
You are an expert math reasoning assistant, specialized in solving multiple-choice problems.
... (content omitted for brevity) ...
FINAL ANSWER: a
"""
# ==============================

DATA_FILE = f"{REPO_ROOT}/ro_mathqa/test.json"

# -------------------------
# Dataset Class
# -------------------------
class MathQADataset(Dataset):
    def __init__(self, json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {json_file}. Please check REPO_ROOT.")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        problem = sample["Problem"]
        choices = sample["Choices"]
        gold = sample["Correct"].lower()

        prompt = (
            system_prompt
            + "\n\nProblem:\n" + problem +
            "\n\nChoices:\n" + choices +
            "\n\n" 
        )
        return {
            "prompt": prompt,
            "gold": gold,
            "problem": problem,
            "choices": choices,
            "idx": idx
        }


# -------------------------
# Extract answer helper
# -------------------------
def extract_letter(text):
    match = re.search(r"FINAL\sANSWER:\s*([a-e])", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).lower() if match else "?"


# -------------------------
# Run model on dataset - OPTIMIZED WITH PROGRESS BAR
# -------------------------
def run_model(model_key, model_name, batch_size=8):
    print(f"\n===== Loading {model_key}: {model_name} =====\n")
    
    dataset = MathQADataset(DATA_FILE)
    if not os.path.exists(DATA_FILE) or not dataset.data:
        print(f"Cannot run model. Data file is missing or empty: {DATA_FILE}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # OPTIMIZATION: Use multiple CPU workers for faster data loading
    num_cpu_workers = os.cpu_count() or 4
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_cpu_workers 
    )

    # NEW: Initialize TQDM progress bar
    total_samples = len(dataset)
    # Calculate total batches: ceiling division (a + b - 1) // b
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    progress_bar = tqdm(loader, 
                        desc=f"Evaluating {model_key}",
                        unit="batch",
                        total=total_batches) 

    results = []
    correct_count = 0
    MAX_GENERATION_TOKENS = 512 

    # Iterate over the progress_bar
    for batch in progress_bar:
        prompts = batch["prompt"]

        enc = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_device = model.device
        enc = {k: v.to(input_device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=MAX_GENERATION_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode newly generated tokens
        input_len = enc['input_ids'].size(1)
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        decoded_full = tokenizer.batch_decode(outputs, skip_special_tokens=True)


        for full_out, model_output, prompt, gold, prob, choices, idx in zip(
            decoded_full,
            decoded_outputs,
            batch["prompt"],
            batch["gold"],
            batch["problem"],
            batch["choices"],
            batch["idx"]
        ):
            model_answer = extract_letter(model_output)
            is_correct = (model_answer == gold)
            correct_count += int(is_correct)

            results.append({
                "idx": int(idx),
                "problem": str(prob),
                "choices": str(choices),
                "gold": str(gold),
                "extracted_answer": str(model_answer),
                "full_reasoning_output": str(model_output),
                "correct": bool(is_correct)
            })

        # Update TQDM postfix to show accuracy and processed count
        progress_bar.set_postfix_str(f"Acc: {100 * correct_count / (len(results) or 1):.2f}% | Done: {len(results)}/{total_samples}")

    accuracy = 100 * correct_count / len(dataset)
    print(f"\n>>> {model_key} accuracy: {accuracy:.2f}%\n")

    out_file = f"{REPO_ROOT}/{model_key}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results (including full reasoning) to {out_file}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    models = {
        "Apertus-8B-Instruct-2509":"swiss-ai/Apertus-8B-Instruct-2509",
        "Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    }
    
    if not os.getenv("REPO_ROOT"):
        print("Warning: REPO_ROOT environment variable is not set. Using default './'. Check your setup.")

    OPTIMAL_BATCH_SIZE = 16 

    for key, name in models.items():
        try:
            run_model(key, name, batch_size=OPTIMAL_BATCH_SIZE)
        except Exception as e:
            print(f"\n--- Error running model {key} ---")
            print(f"An error occurred: {e}")
            print("Skipping to next model.")
            
        torch.cuda.empty_cache()