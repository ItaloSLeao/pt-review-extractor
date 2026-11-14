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
