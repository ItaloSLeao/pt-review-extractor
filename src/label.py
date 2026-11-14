"""Etapa 2: rotula com um modelo professor (Qwen2.5-7B-Instruct AWQ) via vLLM em 1x T4.

uso: python src/label.py --data data
Processa em blocos e salva o progresso em {split}.progress, então pode ser interrompido e retomado.
Gera {split}_labeled.jsonl e test_review.csv (para revisão manual do conjunto de teste).
"""
import argparse
import json
import os

import pandas as pd
from vllm import LLM, SamplingParams

from common import ASPECTS, build_messages, parse_output, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
p.add_argument("--chunk", type=int, default=500)
a = p.parse_args()

llm = LLM(
    model=a.teacher,
    quantization="awq",
    dtype="half",
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=1536,
    gpu_memory_utilization=0.85,
    max_num_seqs=64,
    seed=0,
)
sp = SamplingParams(temperature=0.0, max_tokens=256)

for split in ["train", "val", "test"]:
    rows = read_jsonl(f"{a.data}/{split}_raw.jsonl")
    out_path = f"{a.data}/{split}_labeled.jsonl"
    prog_path = f"{a.data}/{split}.progress"

    done = 0
    if os.path.exists(prog_path) and os.path.exists(out_path):
        with open(prog_path) as f:
            done = int(f.read().strip() or 0)
    else:
        open(out_path, "w").close()  # recomeça do zero

    for start in range(done, len(rows), a.chunk):
        chunk = rows[start : start + a.chunk]
        outs = llm.chat([build_messages(r["text"]) for r in chunk], sp)
        with open(out_path, "a", encoding="utf-8") as f:
            for r, o in zip(chunk, outs):
                label = parse_output(o.outputs[0].text)
                if label is not None:
                    f.write(json.dumps({**r, "label": label}, ensure_ascii=False) + "\n")
        with open(prog_path, "w") as f:
            f.write(str(start + len(chunk)))
        print(f"{split}: {start + len(chunk)}/{len(rows)} processados")

    print(f"{split}: {len(read_jsonl(out_path))}/{len(rows)} rótulos válidos")

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
