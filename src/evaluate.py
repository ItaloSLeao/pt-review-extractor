"""Etapa 4: avalia um modelo (base ou base+adapter) no teste revisado à mão.

uso:
  python src/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --name base-1.5b
  python src/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/qwen1.5b --name ft-1.5b
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ASPECTS, build_messages, parse_output

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter")
p.add_argument("--name", required=True)
p.add_argument("--gold", default="data/test_gold.csv")
p.add_argument("--bs", type=int, default=16)
a = p.parse_args()

gold = pd.read_csv(a.gold, keep_default_na=False)
tok = AutoTokenizer.from_pretrained(a.adapter or a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map={"": 0})
if a.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()

preds, raws, t0 = [], [], time.time()
for i in range(0, len(gold), a.bs):
    batch = gold.text.iloc[i : i + a.bs].tolist()
    prompts = [tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True) for t in batch]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
    raws += texts
    preds += [parse_output(t) for t in texts]
elapsed = time.time() - t0
