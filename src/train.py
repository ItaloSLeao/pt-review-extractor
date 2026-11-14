"""Etapa 3: fine-tuning QLoRA (1x T4, fp16). Loss apenas na resposta (formato prompt/completion).

uso: python src/train.py --model Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen1.5b
"""
import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common import build_messages, read_jsonl, to_json

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--data", default="data")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--r", type=int, default=16)
a = p.parse_args()


def to_ds(split):
    rows = read_jsonl(f"{a.data}/{split}_labeled.jsonl")
    return Dataset.from_list(
        [
            {"prompt": build_messages(r["text"]), "completion": [{"role": "assistant", "content": to_json(r["label"])}]}
            for r in rows
        ]
    )


tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(
    a.model,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    ),
    torch_dtype=torch.float16,
    device_map={"": 0},
)

cfg = SFTConfig(
    output_dir=a.out,
    num_train_epochs=a.epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    gradient_checkpointing=True,
    max_length=768,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="no",
    report_to="none",
    seed=42,
)
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=to_ds("train"),
    eval_dataset=to_ds("val"),
    processing_class=tok,
    peft_config=LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
trainer.save_model(a.out)
tok.save_pretrained(a.out)

import json

with open(f"{a.out}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=1)
