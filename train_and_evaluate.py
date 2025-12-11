import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, DataLoader
import os
import re
from tqdm import tqdm
import math # Needed for training progress tracking

# Set environment variable (assuming REPO_ROOT is correctly set in your environment)
REPO_ROOT = os.getenv("REPO_ROOT", ".") # Defaulting to '.' if not set

# --- FILE PATHS ---
TRAIN_FILE = f"{REPO_ROOT}/ro_mathqa/train.json" # New file path for training
EVAL_FILE = f"{REPO_ROOT}/ro_mathqa/test.json"   # Renamed for clarity in context
OUTPUT_DIR = f"{REPO_ROOT}/sft_output"          # Directory to save the trained model
# ------------------

# === SYSTEM PROMPT FOR BOTH TRAINING AND EVALUATION ===
# The Rationale (gold answer + reasoning) will follow this structure during training.
SYSTEM_PROMPT = """
You are an expert math reasoning assistant, specialized in solving multiple-choice problems.

INPUT FORMAT
- The input contains:
    "Problem": the text of the math problem.
    "Choices": a single string containing all answer options labeled a, b, c, d, e.

YOUR TASK
- Carefully read the problem and choices.
- Output your full, step-by-step reasoning and calculations first.
- At the end, output the chosen letter on a **new, separate line**, prefixed with the exact phrase: **'FINAL ANSWER:'**
- The final output line MUST contain only the prefix and the chosen single letter.

MATH AND SYMBOLS
- Focus on reasoning correctly without altering numeric or symbolic content.

OUTPUT FORMAT
1. **Full Reasoning/Explanation** (multiple lines)
2. **FINAL ANSWER:** followed by the single lowercase letter (a, b, c, d, or e)
"""
# =======================================================


# -------------------------
# 1. NEW: Training Dataset Class
# -------------------------
class MathQATrainDataset(Dataset):
    """
    Loads data and formats it for Supervised Fine-Tuning (SFT).
    The format required is: Prompt + Rationale + Final Answer.
    """
    def __init__(self, json_file, system_prompt):
        with open(json_file, "r", encoding="utf-8") as f:
            # Note: We load the whole file for training
            self.data = json.load(f)
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # NOTE: Assumes the 'Rationale' field contains the step-by-step thinking
        # and the 'Correct' field is the final letter.
        
        # 1. Construct the User Prompt
        user_prompt = (
            self.system_prompt
            + "\n\nProblem:\n" + sample["Problem"] +
            "\n\nChoices:\n" + sample["Choices"] +
            "\n\n" # Ready for model response
        )
        
        # 2. Construct the Model Target Response (Rationale + Final Answer)
        # We parse the Rationale and ensure the final answer format is consistent.
        # Rationale usually contains the reasoning (like the example in the original prompt)
        rationale_text = sample.get("Rationale", "Nu este furnizată o raționare.")
        
        target_response = (
            rationale_text.strip() + 
            f"\n\nFINAL ANSWER: {sample['Correct'].lower()}"
        )
        
        # 3. Full text for the tokenizer
        full_text = user_prompt + target_response
        
        return {"text": full_text}

# -------------------------
# 2. OLD: Evaluation Dataset Class (Modified to use shared prompt)
# -------------------------
class MathQAEvalDataset(Dataset):
    """Dataset class for evaluation (inference)"""
    def __init__(self, json_file, system_prompt):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        problem = sample["Problem"]
        choices = sample["Choices"]
        gold = sample["Correct"].lower()

        prompt = (
            self.system_prompt
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
# Extract answer helper (Remains the same)
# -------------------------
def extract_letter(text):
    match = re.search(r"FINAL\sANSWER:\s*([a-e])", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).lower() if match else "?"


# -------------------------
# NEW: SFT Training Function
# -------------------------

def train_model(model_name, output_dir, train_file, num_train_epochs=3, batch_size=8):
    # Note: Increased default batch_size to 8 here to show optimization effect
    print(f"\n===== Starting SFT Training for: {model_name} =====\n", flush=True)
    
    # NOTE: Need the datasets library for this to work
    from datasets import Dataset as HFDataset 
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading with device_map="auto" already enables multi-GPU use
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    model.resize_token_embeddings(len(tokenizer)) 

    # 2. Load and Prepare Dataset 
    train_dataset = MathQATrainDataset(train_file, SYSTEM_PROMPT)
    formatted_texts = [train_dataset[i]['text'] for i in range(len(train_dataset))]
    hf_dataset = HFDataset.from_dict({'text': formatted_texts})
    
    # OPTIMIZATION: Max length reduced slightly if possible, to 768 tokens (was 1024)
    # Adjust this based on the actual length of your longest examples.
    MAX_SEQ_LENGTH = 768 

    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, # OPTIMIZATION: Shorter sequence length
            padding="max_length", 
            return_tensors='pt'   
        )
        tokenized_output["labels"] = tokenized_output["input_ids"]
        return {k: v.squeeze().tolist() for k, v in tokenized_output.items()}

    tokenized_datasets = hf_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=hf_dataset.column_names
    )
    
    # 3. Define Training Arguments (Optimized)
    
    # Original effective batch size was 8 (4 * 2)
    # New effective batch size is 16 (8 * 2) or 32 (16 * 2) if you have VRAM
    PER_DEVICE_BATCH = 16 # OPTIMIZATION: Increased from 4 to 8
    GRADIENT_ACCUMULATION = 2 # Adjusted for new per_device_batch, keeps effective batch size high

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION, 
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=10,
        fp16=False, 
        bf16=True,  # ESSENTIAL for speed on modern GPUs
        optim="adamw_torch",
        report_to="none", 
        remove_unused_columns=False,
        # OPTIMIZATION: Key to saving VRAM and allowing larger batch size
        gradient_checkpointing=True, 
    )
    
    # 4. Data collator and Trainer setup (as before)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Must activate gradient checkpointing on the model after the Trainer is initialized
    if training_args.gradient_checkpointing:
        model.config.use_cache = False # Disable cache when using gradient checkpointing
        
    # 5. Start Training
    print("Starting training...", flush=True)
    trainer.train()

    # 6. Save the final model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Training complete. Model saved to: {output_dir}", flush=True)
    return output_dir
# -------------------------
# 3. OLD: Evaluation Function (Renamed and simplified)
# -------------------------
def evaluate_model(model_key, model_path, batch_size=16):
    print(f"\n===== Re-Evaluating {model_key} from {model_path} =====\n", flush=True)
    
    dataset = MathQAEvalDataset(EVAL_FILE, SYSTEM_PROMPT)
    if not dataset.data:
        print("Cannot run evaluation. Dataset is empty.", flush=True)
        return

    # Load tokenizer and model from the saved path
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # OPTIMIZATION: DataLoader setup remains the same
    num_cpu_workers = os.cpu_count() or 4
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_cpu_workers 
    )

    # TQDM progress bar setup
    total_samples = len(dataset)
    total_batches = (total_samples + batch_size - 1) // batch_size
    progress_bar = tqdm(loader, desc=f"Evaluating {model_key}", unit="batch", total=total_batches) 

    results = []
    correct_count = 0
    MAX_GENERATION_TOKENS = 512 

    # Inference loop (remains the same)
    for batch in progress_bar:
        prompts = batch["prompt"]

        enc = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
        input_device = model.device
        enc = {k: v.to(input_device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=MAX_GENERATION_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = enc['input_ids'].size(1)
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for model_output, gold, idx in zip(decoded_outputs, batch["gold"], batch["idx"]):
            model_answer = extract_letter(model_output)
            is_correct = (model_answer == gold)
            correct_count += int(is_correct)

            results.append({
                "idx": int(idx),
                "gold": str(gold),
                "extracted_answer": str(model_answer),
                "full_reasoning_output": str(model_output),
                "correct": bool(is_correct)
            })

        progress_bar.set_postfix_str(f"Acc: {100 * correct_count / (len(results) or 1):.2f}% | Done: {len(results)}/{total_samples}")

    accuracy = 100 * correct_count / total_samples
    print(f"\n>>> {model_key} Fine-Tuned Accuracy: {accuracy:.2f}%\n", flush=True)

    out_file = f"{model_path}/fine_tuned_{model_key}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation results to {out_file}", flush=True)


# -------------------------
# Main Execution Flow
# -------------------------
if __name__ == "__main__":
    
    # NOTE: You MUST have the 'datasets' library installed for the Trainer to work.
    # pip install datasets

    if not os.getenv("REPO_ROOT"):
        print("Warning: REPO_ROOT environment variable is not set. Using default './'. Check your setup.", flush=True)
    
    models_to_train = {
        "Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    }
    
    OPTIMAL_BATCH_SIZE = 4 # Use a smaller batch size for training (requires VRAM)
    NUM_EPOCHS = 3         # Number of training passes

    for key, name in models_to_train.items():
        MODEL_OUTPUT_PATH = f"{OUTPUT_DIR}/{key}_sft"
        
        # --- 1. Train and Save ---
        try:
            # We use a lower batch size for training than for inference
            final_model_path = train_model(
                model_name=name,
                output_dir=MODEL_OUTPUT_PATH,
                train_file=TRAIN_FILE,
                num_train_epochs=NUM_EPOCHS,
                batch_size=OPTIMAL_BATCH_SIZE
            )
        except Exception as e:
            print(f"\n--- FATAL ERROR during SFT Training for {key} ---", flush=True)
            print(f"Error: {e}", flush=True)
            print("Skipping evaluation for this model.", flush=True)
            final_model_path = None
        finally:
            torch.cuda.empty_cache()

        # --- 2. Re-Evaluate ---
        if final_model_path:
            # We use a higher batch size for evaluation (inference) than for training
            EVAL_BATCH_SIZE = 32 
            try:
                evaluate_model(key, final_model_path, batch_size=EVAL_BATCH_SIZE)
            except Exception as e:
                print(f"\n--- ERROR during Evaluation for {key} ---", flush=True)
                print(f"Error: {e}", flush=True)
                
            torch.cuda.empty_cache()