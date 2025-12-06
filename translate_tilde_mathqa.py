import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset


REPO_ROOT = os.getenv("REPO_ROOT")
OUTPUT_MATHQA_ROOT = REPO_ROOT + "/ro_mathqa/"


system_prompt = """You are an expert translator from English to Romanian, specialized in math multiple-choice problems.

INPUT FORMAT
- The input is a JSON object with keys: "Problem", "Rationale" and "Choices"
- "Problem": the text of the math problem.
- "Rationale": a step-by-step explanation.
- "Choices": a single string containing all answer options (a, b, c, …).

YOUR TASK
- Translate "Problem" and "Rationale" into natural, clear Romanian.
- Translate the words in "Choices" into Romanian, but keep option letters and math expressions aligned with the original.

MATH AND SYMBOLS
- Do NOT change or recompute any numbers.
- Keep all math expressions and symbols exactly as in the original:
  - Fractions like "4 / 5", "10 / 20"
  - Arithmetic like "10 + 5", "3 * 4", "2 ^ 3"
  - Percentages like "10 %", "240 %"
  - Constants and symbols like "π", "√", ">", "<", "="
  - Units like "kg", "cm", "cm 2", "m / s"
  - Currency amounts like "$200", "rs . 400", "Rs . 823"
- You may translate words like "dollars", "rupees", "kilograms", "centimeters" into Romanian, but keep the numeric parts and symbols exactly as in the original.

STYLE
- Use correct, clear Romanian, as if explaining to a high-school student.
- Use natural phrasing, not word-for-word translation.
- Make sure subject-verb agreement and gender/number agreement are correct.
- Keep the logic and order of the reasoning; you may slightly rephrase for naturalness.

OUTPUT FORMAT
- Output must be a valid JSON object with the SAME KEYS: "Problem", "Rationale" and "Choices".
- Do NOT add any extra keys.
- Do NOT add explanations, comments, or text outside this JSON.

EXAMPLE 1

Input:
{
  "Problem": "There are 10 girls and 20 boys in a classroom. What is the ratio of girls to boys?",
  "Rationale": "If girls is 10 and boys is 20, then 10 / 20 = 1 / 2. So the ratio of girls to boys is 1 / 2. Answer: a",
  "Choices": "a ) 1 / 2 , b ) 1 / 3 , c ) 1 / 5 , d ) 10 / 30 , e ) 2 / 5"
}

Output:
{
  "Problem": "Într-o clasă sunt 10 fete și 20 de băieți. Care este raportul dintre fete și băieți?",
  "Rationale": "Dacă numărul fetelor este 10 și al băieților este 20, atunci 10 / 20 = 1 / 2. Deci raportul dintre fete și băieți este 1 / 2. Răspuns: a",
  "Choices": "a ) 1 / 2 , b ) 1 / 3 , c ) 1 / 5 , d ) 10 / 30 , e ) 2 / 5"
}

EXAMPLE 2

Input:
{
  "Problem": "An article is bought for rs . 823 and sold for rs . 1000. Find the gain percent ?",
  "Rationale": "Gain = 1000 - 823 = 177. Gain percent = 177 / 823 * 100 ≈ 21.5 % . Answer : b",
  "Choices": "a ) 21.4 % , b ) 21.5 % , c ) 21.6 % , d ) 21.7 % , e ) 21.8 %"
}

Output:
{
  "Problem": "Un articol este cumpărat cu rs . 823 și vândut cu rs . 1000. Găsește procentul de profit.",
  "Rationale": "Profitul este 1000 - 823 = 177. Procentul de profit este 177 / 823 * 100 ≈ 21.5 % . Răspuns : b",
  "Choices": "a ) 21.4 % , b ) 21.5 % , c ) 21.6 % , d ) 21.7 % , e ) 21.8 %"
}
"""


# Check GPU bindings
def check_gpu_bindings():
    print("torch.cuda.device_count() =", torch.cuda.device_count(), flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)


def run_model(model_key, model_name, dataset, split):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=(False if model_key == "TildeOpen-30b" else True),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Loaded model for {split=}", flush=True)

    translations = []

    def dump_to_file():
        out_file = os.path.join(OUTPUT_MATHQA_ROOT, f"{split}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)

    limit = len(dataset[split])
    batch_size = 16

    for start_idx in range(0, limit, batch_size):
        end_idx = min(start_idx + batch_size, limit)

        prompts = []
        for idx in range(start_idx, end_idx):
            raw = dataset[split][idx]
            sample = {
                "Problem": raw["Problem"],
                "Rationale": raw["Rationale"],
                "Choices": raw["options"],
            }
            user_content = json.dumps(sample, ensure_ascii=False)
            prompt = (
                system_prompt
                + f"\n\nNow translate this input:\n{user_content}\n\n"
                  "Output (JSON only, no extra sentences):\n"
            )
            prompts.append(prompt)

        bsz = end_idx - start_idx
        outputs = gen(
            prompts,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            batch_size=bsz,
        )

        for prompt, output, idx in zip(prompts, outputs, range(start_idx, end_idx)):
            full_output = output[0]["generated_text"]
            response_str = full_output[len(prompt):].strip()

            start = response_str.find("{")
            end = response_str.rfind("}")
            if start == -1 or end == -1 or end <= start:
                print(f"[{model_key}] Failed to parse JSON for idx={idx}:\n{response_str}\n")
                continue

            response_str = response_str[start : end + 1]

            try:
                translated = json.loads(response_str)
                translated["Correct"] = dataset[split][idx]["correct"]
            except Exception as e:
                print(f"[{model_key}] JSON decode error at idx={idx}: {e}\n{response_str}\n")
                continue

            translations.append(translated)
            print(f"Done {idx + 1}/{limit} for {split=}", flush=True)

            if idx % 50 == 0:
                dump_to_file()

    dump_to_file()
    print(f"Done translations for {split=}", flush=True)


if __name__ == "__main__":
    print(f"{REPO_ROOT=}", flush=True)

    check_gpu_bindings()

    mathqa = load_dataset("regisss/math_qa")
    print("Loaded MathQA", flush=True)

    for split in ["train", "validation", "test"]:
        run_model("TildeOpen-30b", "TildeAI/TildeOpen-30b", mathqa, split)
        torch.cuda.empty_cache()
