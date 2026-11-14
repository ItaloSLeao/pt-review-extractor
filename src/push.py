"""Etapa 5: publica dataset e modelo (merged, fp16) no Hugging Face Hub.

uso: python src/push.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/qwen1.5b --repo SEU_USER/qwen2.5-1.5b-avaliacoes-ptbr
     python src/push.py --dataset_repo SEU_USER/avaliacoes-ptbr-estruturadas
"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--base")
p.add_argument("--adapter")
p.add_argument("--repo")
p.add_argument("--dataset_repo")
p.add_argument("--data", default="data")
a = p.parse_args()

if a.dataset_repo:
    import pandas as pd
    from datasets import Dataset, DatasetDict

    from common import ASPECTS, read_jsonl

    def flat(split):
        return Dataset.from_list([{"id": r["id"], "text": r["text"], "label": r["label"]} for r in read_jsonl(f"{a.data}/{split}_labeled.jsonl")])

    gold = pd.read_csv(f"{a.data}/test_gold.csv", keep_default_na=False)
    test = Dataset.from_list(
        [
            {
                "id": r.id,
                "text": r.text,
                "label": {
                    "sentimento": r.sentimento,
                    "aspectos": {k: (getattr(r, k) or None) for k in ASPECTS},
                    "problema_principal": r.problema_principal or None,
                    "requer_acao": str(r.requer_acao).strip().lower() == "true",
                },
            }
            for r in gold.itertuples()
        ]
    )
    DatasetDict({"train": flat("train"), "validation": flat("val"), "test": test}).push_to_hub(a.dataset_repo)

if a.repo:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.push_to_hub(a.repo)
    AutoTokenizer.from_pretrained(a.adapter).push_to_hub(a.repo)
