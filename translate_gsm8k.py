from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# NLLB 600M
nllb_translate = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="ron_Latn",
    max_length=256,
    device=0,
    do_sample=False
)

# LLama-3 8B
llama3_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_id)
llama3_pipe = pipeline(
    "text-generation",
    model=llama3_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    max_new_tokens=256,
    device=1,
    do_sample=False,
    temperature=1.0,
    top_p=1.0
)


def nllb(sample):
    return {
        'question': nllb_translate(sample['question'])[0]['translation_text'],
        'answer': nllb_translate(sample['answer'])[0]['translation_text']
    }


def llama(sample):
    base_prompt = (
        "You are a translator from English to Romanian. You translate math problems with questions and answers. "
        "You will get such questions and answers to translate. "
        "Use natural, correct Romanian. Keep the tone simple and clear. "
        "Preserve numbers and mathematical expressions. Keep these expressions the same as in the english version. "
        "Output only the translation with no explanation, comments or extra sentences. Do not repeat tokens."
    )

    split_token = "Traducere în română:"

    question_prompt = base_prompt + (
        f"Question to translate: {sample['question']}\n"
        f"{split_token}"
    )

    answer_prompt = base_prompt + (
        f"Answer to translate: {sample['answer']}\n"
        f"{split_token}"
    )

    question_out = llama3_pipe(question_prompt, eos_token_id=llama3_pipe.tokenizer.eos_token_id, pad_token_id=llama3_pipe.tokenizer.pad_token_id)[0]["generated_text"]
    question_translated = question_out.split(split_token)[-1].strip()

    answer_out = llama3_pipe(answer_prompt, eos_token_id=llama3_pipe.tokenizer.eos_token_id, pad_token_id=llama3_pipe.tokenizer.pad_token_id)[0]["generated_text"]
    answer_translated = answer_out.split(split_token)[-1].strip()

    return {
        'question': question_translated,
        'answer': answer_translated
    }


def print_translations(idx, sample, nllb_out, llama_out):
    print(f"\n===== SAMPLE {idx} =====")
    print("EN QUESTION:", sample["question"])
    print("EN ANSWER:  ", sample["answer"])

    print("\nNLLB QUESTION:", nllb_out["question"])
    print("NLLB ANSWER:  ", nllb_out["answer"])

    print("\nLLAMA QUESTION:", llama_out["question"])
    print("LLAMA ANSWER:  ", llama_out["answer"])
    print("=" * 100)


if __name__ == "__main__":
    dataset = load_dataset("openai/gsm8k", name="main", split="train")
    print(nllb_translate.model.device)
    print(llama3_pipe.model.device)

    for i in range(50):
        print_translations(i, dataset[i], nllb(dataset[i]), llama(dataset[i]))
