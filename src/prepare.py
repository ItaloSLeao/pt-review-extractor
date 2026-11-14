"""Etapa 1: amostra avaliações da Olist e separa train/val/test.

uso: python src/prepare.py --csv /kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv --out data
"""
import argparse
import os

import pandas as pd

from common import write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", default="data")
p.add_argument("--n_train", type=int, default=5000)
p.add_argument("--n_val", type=int, default=300)
p.add_argument("--n_test", type=int, default=300)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

df = pd.read_csv(a.csv)
df["text"] = (
    df["review_comment_title"].fillna("").str.strip() + ". " + df["review_comment_message"].fillna("").str.strip()
).str.strip(". ").str.replace(r"\s+", " ", regex=True)
df = df[df["review_comment_message"].notna() & (df["text"].str.len() >= 20)]
df = df.drop_duplicates("text")
print(f"{len(df)} avaliações com texto")
