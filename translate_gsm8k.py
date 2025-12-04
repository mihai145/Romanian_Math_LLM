from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ==============================
# LLM PIPELINE SETUP
# ==============================
def init_llm_pipeline(model_id, device, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        max_new_tokens=max_tokens,
        device=device,
        do_sample=False,
        temperature=1.0,
        top_p=1.0
    )


# Initialize models on separate GPUs
eurollm_pipe = init_llm_pipeline("utter-project/EuroLLM-9B-Instruct", device=0)
gemma_pipe = init_llm_pipeline("google/gemma-3-27b-it", device=1)
# llama70_pipe = init_llm_pipeline("meta-llama/Llama-4-Scout-17B-16E", device=2)


# ==============================
# PROMPT FUNCTION (raw output, old style)
# ==============================
def make_translation_prompt(sample, target_language="Romanian"):
    return f""" ## Instructions Imagine you're part of a team at an international education center that's revamping its maths exams for a global audience. Your job is to translate an English question and its answer options into {target_language} so that students from {target_language} schools can be evaluated too. Just provide the final translation—leave out any extra comments or explanations. Use language which is authentic for {target_language} natives. Remember to keep the answer options connected to the question, using the same format as the original (a list for multiple choices or plain text for a single answer). This is important - please make sure that the original format of question and answers is preserved - some formulas/expressions are highlighted by <<expression>> and the final answer is highlighted by ####. Please double check that your final answer is included in the final translation (highlighted by ####). ## Original text Here's what you need to work on: {{ "Original_question": \"\"\"{sample['question']}\"\"\", "Original_answer": \"\"\"{sample['answer']}\"\"\" }} ## Output instructions Fill the fields below with the correct translation into {target_language}. Do not leave them empty. Do not repeat the English text. Do not rewrite the JSON keys. Fill in ONLY the {target_language} text between the quotation marks. Return ONLY this completed JSON object and nothing else: {{ "Question": \"\"\"\n\"\"\", "Answer": \"\"\"\n\"\"\" }} """

# ==============================
# TRANSLATION FUNCTION
# ==============================
def translate_with_llm(sample, llm_pipe, target_language="Romanian"):
    prompt = make_translation_prompt(sample, target_language)
    output = llm_pipe(prompt)[0]["generated_text"]
    return output.strip()


# ==============================
# PRINTING FUNCTION
# ==============================
def print_translations(idx, sample, translations):
    print(f"\n===== SAMPLE {idx} =====")
    print("EN QUESTION:", sample["question"])
    print("EN ANSWER:  ", sample["answer"])
    for name, out in translations.items():
        print(f"\n{name} TRANSLATION:\n{out}")
    print("=" * 100)


# ==============================
# MAIN LOOP
# ==============================
if __name__ == "__main__":
    dataset = load_dataset("openai/gsm8k", name="main", split="train")

    for i in range(5):  # testing with first 5 examples
        sample = dataset[i]
        translations = {
            "EuroLLM": translate_with_llm(sample, eurollm_pipe),
            "Gemma-IT": translate_with_llm(sample, gemma_pipe),
            #"Llama-70B": translate_with_llm(sample, llama70_pipe)
        }
        print_translations(i, sample, translations)
