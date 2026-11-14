"""Etapa 2: rotula com um modelo professor (Qwen2.5-7B-Instruct AWQ) via vLLM em 2x T4.

uso: python src/label.py --data data
Gera {split}_labeled.jsonl e test_review.csv (para revisão manual do conjunto de teste).
"""
import argparse

import pandas as pd
from vllm import LLM, SamplingParams

from common import ASPECTS, build_messages, parse_output, read_jsonl, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
a = p.parse_args()

llm = LLM(model=a.teacher, quantization="awq", dtype="half", tensor_parallel_size=1, max_model_len=2048, seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=256)

for split in ["train", "val", "test"]:
    rows = read_jsonl(f"{a.data}/{split}_raw.jsonl")
    outs = llm.chat([build_messages(r["text"]) for r in rows], sp)
    kept = []
    for r, o in zip(rows, outs):
        label = parse_output(o.outputs[0].text)
        if label is not None:
            kept.append({**r, "label": label})
    write_jsonl(f"{a.data}/{split}_labeled.jsonl", kept)
    print(f"{split}: {len(kept)}/{len(rows)} rótulos válidos")

# CSV plano para corrigir o teste à mão (Google Sheets/LibreOffice).
test = read_jsonl(f"{a.data}/test_labeled.jsonl")
flat = [
    {
        "id": r["id"],
        "text": r["text"],
        "sentimento": r["label"]["sentimento"],
        **{a_: r["label"]["aspectos"][a_] or "" for a_ in ASPECTS},
        "problema_principal": r["label"]["problema_principal"] or "",
        "requer_acao": r["label"]["requer_acao"],
        "revisado": "",
    }
    for r in test
]
pd.DataFrame(flat).to_csv(f"{a.data}/test_review.csv", index=False)
print("Revise data/test_review.csv e salve como data/test_gold.csv")
