import os
import sys
import json
from pathlib import Path
import re

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


DATASET_PATHS = {
    "ro_gsm8k": "/iopsstor/scratch/cscs/moprea/Romanian_Math_LLM/ro_gsm8k/train.json",
    "ro_mathqa": "/iopsstor/scratch/cscs/moprea/Romanian_Math_LLM/ro_mathqa/train.json"
}

dbg_printed, dbg_to_print = 0, 10

def model_tag(s: str) -> str:
    return s.replace("/", "__").replace(":", "__")


def gsm8k_prompt_and_target(ex: dict):
    global dbg_printed

    q = ex["Question"].strip()
    ans = ex["Answer"].split("####")[-1].strip()

    prompt = (
        "Rezolvă problema de mai jos.\n"
        "Returnează DOAR răspunsul final ca NUMĂR (fără text).\n"
        f"Întrebare: {q}\n"
        "Răspuns:\n"
    )
    target = ans

    if dbg_printed < dbg_to_print:
        dbg_printed += 1
        print(prompt)
        print(target)

    return prompt, target


def mathqa_prompt_and_target(ex: dict):
    global dbg_printed

    p = ex["Problem"].strip()
    ch = ex["Choices"].strip()
    correct = str(ex["Correct"]).strip().lower()

    choice_text = ""
    marks = list(re.finditer(r"\b([a-e])\s*\)\s*", ch.lower()))
    if marks:
        mapping = {}
        for i, m in enumerate(marks):
            letter = m.group(1)
            start = m.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(ch)
            txt = ch[start:end].strip().rstrip(",;").strip()
            mapping[letter] = txt
        choice_text = mapping.get(correct, "")

    prompt = (
        "Alege varianta corectă.\n"
        "Returnează DOAR în formatul: <litera>) <textul opțiunii>\n"
        f"Întrebare: {p}\n"
        f"Opțiuni: {ch}\n"
        "Răspuns:\n"
    )
    target = f"{correct}) {choice_text}" if choice_text else correct

    if dbg_printed < dbg_to_print:
        dbg_printed += 1
        print(prompt)
        print(target)

    return prompt, target


def main():
    if len(sys.argv) != 3:
        print("Usage: python train_sft_simple.py <MODEL> <DATASET>")
        sys.exit(2)

    model_id = sys.argv[1]
    dataset = sys.argv[2]

    out_root = os.environ.get("SFT_OUT_ROOT", "/capstor/scratch/cscs/moprea/finetuned/")
    max_steps = int(os.environ.get("MAX_STEPS"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN"))
    lr = float(os.environ.get("LR"))
    per_device_batch = int(os.environ.get("BATCH"))
    grad_accum = int(os.environ.get("GRAD_ACCUM"))

    out_dir = Path(out_root) / model_tag(model_id) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model={model_id}")
    print(f"[INFO] dataset={dataset}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] max_steps={max_steps} max_seq_len={max_seq_len} lr={lr} batch={per_device_batch} grad_accum={grad_accum}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(ex):
        if dataset == "ro_gsm8k":
            prompt, target = gsm8k_prompt_and_target(ex)
        else:
            prompt, target = mathqa_prompt_and_target(ex)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    ds = load_dataset("json", data_files={"train": DATASET_PATHS[dataset]})["train"]
    if dataset == "ro_gsm8k":
        ds = ds.filter(lambda x: x.get("Question") and x.get("Answer"), load_from_cache_file=False)
    else:
        ds = ds.filter(lambda x: x.get("Problem") and x.get("Choices") and x.get("Correct"), load_from_cache_file=False)

    ds = ds.map(format_example, remove_columns=ds.column_names, load_from_cache_file=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"[INFO] local_rank={local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
        load_in_4bit=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=max_steps,
        learning_rate=lr,
        warmup_ratio=0.03,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,
        max_length=max_seq_len,
        packing=False,
        padding_free=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        try:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
            tokenizer.save_pretrained(str(out_dir / "merged"))
            print(f"[DONE] Saved merged model to {out_dir/'merged'}")
        except Exception as e:
            print(f"[WARN] Could not merge adapters (still saved adapter). Error: {e}")

        (out_dir / "run_info.txt").write_text(
            f"model={model_id}\ntrain_json={dataset}\n"
            f"max_steps={max_steps}\nmax_seq_len={max_seq_len}\nlr={lr}\nbatch={per_device_batch}\ngrad_accum={grad_accum}\n",
            encoding="utf-8",
        )

        print(f"[DONE] Saved to {out_dir}")


if __name__ == "__main__":
    main()
