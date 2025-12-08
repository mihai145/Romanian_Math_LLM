import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import os
import re


REPO_ROOT = os.getenv("REPO_ROOT")

system_prompt = """
You are an expert math reasoning assistant, specialized in solving multiple-choice problems.

INPUT FORMAT
- The input contains:
    "Problem": the text of the math problem.
    "Choices": a single string containing all answer options labeled a, b, c, d, e.
- Each problem may include numbers, arithmetic expressions, percentages, units, or symbols.

YOUR TASK
- Carefully read the problem and choices.
- Think step by step about the solution internally.
- At the end, choose the best answer and output ONLY a single letter: a, b, c, d, or e.
- Do NOT output any explanations, reasoning, or restatements of the question.
- Do NOT output anything except the chosen letter.

MATH AND SYMBOLS
- Do NOT recompute or change any numbers.
- Keep all math expressions exactly as written, including:
    - Fractions: "4 / 5", "10 / 20"
    - Arithmetic: "3 + 5", "2 * 4", "2 ^ 3"
    - Percentages: "10 %", "240 %"
    - Constants and symbols: "π", "√", ">", "<", "="
    - Units: "kg", "cm", "m / s"
    - Currency amounts: "$200", "rs . 400", "Rs . 823"
- Focus on reasoning correctly without altering numeric or symbolic content.

STYLE
- Use clear, concise, and correct reasoning internally.
- Ensure all logical deductions follow from the problem.
- Subject-verb and gender/number agreement do not apply to the output letter, but must be used internally if reasoning is verbalized.

OUTPUT
- Output only a single lowercase letter corresponding to the answer (a, b, c, d, or e).

EXAMPLE:

GIVEN INPUT:

"Problem": "câștigul bancherului pentru o anumită sumă datorată peste 3 ani la 10 % pe an este rs. 36. care este valoarea actuală?",
"Rationale": "\"explicație : t = 3 ani r = 10 % td = ( bg × 100 ) / tr = ( 36 × 100 ) / ( 3 × 10 ) = 12 × 10 = rs. 120 td = ( pw × tr ) / 100 ⇒ 120 = ( pw × 3 × 10 ) / 100 ⇒ 1200 = pw × 3 pw = 1200 / 3 = rs. 400 răspuns : opțiunea a\"",
"Choices": "a ) rs. 400, b ) rs. 300, c ) rs. 500, d ) rs. 350, e ) niciuna dintre acestea",

REQUIRED OUTPUT:
"a"

"""


DATA_FILE = f"{REPO_ROOT}/ro_mathqa/test.json"

# -------------------------
# Dataset Class
# -------------------------
class MathQADataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            self.data = self.data[:5]

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
            "\n\nAnswer (letter only):\n"
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
    match = re.search(r"\b([a-e])\b", text.lower())
    return match.group(1) if match else "?"


# -------------------------
# Run model on dataset
# -------------------------
def run_model(model_key, model_name, batch_size=8):
    print(f"\n===== Loading {model_key}: {model_name} =====\n")

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

    dataset = MathQADataset(DATA_FILE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    correct_count = 0

    for batch in loader:
        prompts = batch["prompt"]

        enc = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for full_out, prompt, gold, prob, choices, idx in zip(
            decoded,
            batch["prompt"],
            batch["gold"],
            batch["problem"],
            batch["choices"],
            batch["idx"]
        ):

            model_output = full_out[len(prompt):].strip()
            model_answer = extract_letter(model_output)

            is_correct = (model_answer == gold)
            correct_count += int(is_correct)

            results.append({
                "idx": int(idx),
                "problem": str(prob),
                "choices": str(choices),
                "gold": str(gold),
                "model_answer": str(model_answer),
                "raw_output": str(model_output),  # ensure string, not tensor
                "correct": bool(is_correct)
            })

            print(f"[{model_key}] Q{idx+1}: model={model_answer}, gold={gold}, ok={is_correct}")

    accuracy = 100 * correct_count / len(dataset)
    print(f"\n>>> {model_key} accuracy: {accuracy:.2f}%\n")

    # Save results
    out_file = f"{REPO_ROOT}/{model_key}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_file}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    models = {
        # tiny models for testing
        # "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "Zephyr-7B": "HuggingFaceH4/zephyr-7b-alpha",

        # your real models (uncomment later)
        "TildeOpen-30b": "TildeAI/TildeOpen-30b",
        "Tower-Plus-72B": "Unbabel/Tower-Plus-72B",
    }

    for key, name in models.items():
        run_model(key, name, batch_size=4)
        torch.cuda.empty_cache()
