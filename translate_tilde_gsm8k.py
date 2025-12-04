import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import os


REPO_ROOT = os.getenv("REPO_ROOT")
OUTPUT_GSM8K_ROOT = REPO_ROOT + "/ro_gsm8k/"

system_prompt="""You are an expert translator from English to Romanian, specialized in math problems. Your task:
    - Translate the given math problem from English to Romanian
    - The input is a JSON object with keys "Question" and "Answer"
    - You must return only a JSON object with the same keys and the content correctly translated into Romanian

    Use correct, clear and simple Romanian, as if explaining to a high-school student.
    Use natural phrasing, not word for word translation.
    Always check agreement between subject and verb.
    Always check gender and number agreement between nouns, adjectives and participles.
    You may slightly reorder the wording if that sounds more natural in Romanian.

    Math notation and special tokens may appear. Do not change anything inside << and >>, or after ####, if they appear.
    For example, keep <<101-13=88>> or #### 10 as is.
    Keep numbers and operations exactly as in the original, do not recompute anything.

    Read the whole problem, question and answer, and make sure the translation is coherent as a whole.
    Keep the structure of the reasoning, but feel free to slightly reorder or reword if that makes it more natural in Romanian.

    Your output must be valid JSON, with the same keys as the original, "Question" and "Answer".
    Do not add explanations, comments or any extra text outside this JSON.

    Here are three examples of the task you should perform.

    Input 1:
    {
        "Question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
        "Answer": "Let S be the amount Alexis paid for the shoes. She spent S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143. She used all but $16 of her budget, so S + 143 = 200 - 16 = 184. Thus, Alexis paid S = 184 - 143 = $<<184-143=41>>41 for the shoes. #### 41"
    }

    Output 1:
    {
        "Question": "Alexis aplică pentru un nou loc de muncă și și-a cumpărat un set de haine business pentru a le purta la interviu. Ea a mers la magazin cu un buget de $200 și a cheltuit $30 pe o cămașă, $46 pe pantaloni de costum, $38 pe un sacou, $11 pe șosete și $18 pe o curea. Ea a cumpărat și o pereche de pantofi, dar a pierdut bonul fiscal pentru aceștia. Ea mai are $16 rămași din bugetul ei. Cât a plătit Alexis pentru pantofi?",
        "Answer": "Fie S suma pe care Alexis a plătit-o pe pantofi. Ea a cheltuit S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143. Ea a folosit tot bugetul mai puțin $16 dolari, deci S + 143 = 200 - 16 = 184. Deci, Alexis a plătit S = 184 - 143 = $<<184-143=41>>41 pentru pantofi. #### 41"
    }

    Input 2:
    {
        "Question": "Artemis is making tea for a party. She knows her mom drinks an 8-ounce cup of tea and uses one ounce of tea. She will use this same ratio for the party. The party has 12 people there and each of them wants a 6-ounce cup of tea. How many ounces of tea does she need?",
        "Answer": "She is making 72 ounces of water because 12 x 6 = <<12*6=72>>72 She needs 9 ounces of tea because 72 / 8 = <<72/8=9>>9 #### 9"
    }

    Output 2:
    {
        "Question": "Artemis pregătește ceai pentru o petrecere. Ea știe că mama ei bea un pahar de 8 uncii și pentru asta folosește o uncie de ceai. Artemis va folosi aceeași rație pentru petrecere. La petrecere vin 12 persoane și fiecare dintre ele vrea un pahar de 6 uncii de ceai. De câte uncii de ceai are Artemis nevoie?",
        "Answer": "Artemis pregătește 72 de uncii de apă pentru că 12 x 6 = <<12*6=72>>72. Ea are nevoie de 9 uncii de ceai pentru că 72 / 8 = <<72/8=9>>9 #### 9"
    }

    Input 3:
    {
        "Question": "Elliott drew a right-angle triangle on his book. It had a base of 4 inches, a height of 3 inches and a certain length of the hypotenuse. What was the length of the perimeter of the triangle that he drew?",
        "Answer": "Since the hypotenuse of a triangle is found by the square root of the base squared plus the height squared. The square of the base of the first triangle is 4*4=<<4*4=16>>16 square inches The square of the height of the first triangle is 3*3=<<3*3=9>>9 square inches. The sum of the squares of the base and the height of the first triangle is 16+9=<<16+9=25>>25 The square root of the sum of the base and the height of the first triangle, which is the hypotenuse of the triangle, is √25=5 inches. Since the perimeter of a triangle is found by adding the sides of the triangle, the base and height of the triangle sum up to 3+4=<<3+4=7>>7 inches If you add the hypotenuse of the triangle the perimeter of the triangle becomes 7+5=<<7+5=12>>12 inches. #### 12"
    }

    Output 3:
    {
        "Question": "Elliott a desenat un triunghi dreptunghic în cartea sa. Triunghiul avea o bază de 4 inci, o înălțime de 3 inci și o anumită lungime a ipotenuzei. Care este lungimea perimetrului triunghiului pe care el l-a desenat?",
        "Answer": "Ipotenuza triunghiului este rădăcina pătrată a sumei pătratelor bazei și înălțimii. Pătratul bazei triunghiului este 4*4=<<4*4=16>>16 inci pătrați. Pătratul înălțimii triunghiului este 3*3=<<3*3=9>>9 inci pătrați. Suma pătratelor bazei și înălțimii este 16+9=<<16+9=25>>25. Rădăcina pătrată a sumei pătratelor bazei și înălțimii triunghiului, care este ipotenuza triunghiului, este √25=5 inci. Deoarece perimetrul triunghiului se calculează adunând lungimile laturilor triunghiului, baza și înălțimea triunghiului însumează 3+4=<<3+4=7>>7 inci. Dacă adunăm ipotenuza triunghiului, perimetrul triunghiului devine 7+5=<<7+5=12>>12 inci. #### 12"
    }
"""

# Check GPU bindings
def check_gpu_bindings():
    print("torch.cuda.device_count() =", torch.cuda.device_count(), flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)


def run_model(model_key, model_name, dataset, split):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=(False if model_key == "TildeOpen-30b" else True))
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
    print(f"Loaded model for {split=}")

    translations = []
    def dump_to_file():
        out_file = f"{OUTPUT_GSM8K_ROOT}/{split}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)

    # limit = min(1000, len(dataset[split]))
    limit = len(dataset[split])
    batch_size = 16

    for start_idx in range(0, limit, batch_size):
        end_idx = min(start_idx + batch_size, limit)

        prompts = []
        for idx in range(start_idx, end_idx):
            raw = dataset[split][idx]
            sample = {
                "Question": raw["question"],
                "Answer": raw["answer"]
            }
            user_content = json.dumps(sample, ensure_ascii=False)
            prompt = system_prompt + f'\n\nNow translate this input:\n{user_content}\n\nOutput (JSON only, no extra sentences):\n'
            prompts.append(prompt)

        bsz = end_idx - start_idx
        outputs = gen(
            prompts,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            batch_size=bsz
        )

        for prompt, output, idx in zip(prompts, outputs, range(start_idx, end_idx)):
            full_output = output[0]["generated_text"]
            response_str = full_output[len(prompt):].strip()

            start = response_str.find("{")
            end = response_str.find("}")
            if start == -1 or end == -1 or end <= start:
                print(f"[{model_key}] Failed to parse JSON:\n{response_str}\n")
                continue

            response_str = response_str[start:end+1]

            try:
                translated = json.loads(response_str)
            except:
                print(f"[{model_key}] Failed to parse JSON:\n{response_str}\n")
                continue

            translations.append(translated)
            print(f'Done {idx+1}/{limit} for {split=}', flush=True)

            if idx % 10 == 0:
                dump_to_file()

    dump_to_file()
    print(f'Done translations for {split=}', flush=True)


if __name__ == "__main__":
    print(f'{REPO_ROOT=}', flush=True)

    check_gpu_bindings()

    gsm8k = load_dataset("gsm8k", "main")
    print("Loaded gsm8k", flush=True)

    for split in ["train", "test"]:
        run_model("TildeOpen-30b", "TildeAI/TildeOpen-30b", gsm8k, split)
        torch.cuda.empty_cache()
