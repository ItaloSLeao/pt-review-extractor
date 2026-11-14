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

# Balanceia por nota: notas baixas são minoria mas são onde está o sinal útil.
n_total = a.n_train + a.n_val + a.n_test
per_score = n_total // 5
parts = [g.sample(min(len(g), per_score), random_state=a.seed) for _, g in df.groupby("review_score")]
s = pd.concat(parts)
if len(s) < n_total:
    rest = df.drop(s.index).sample(n_total - len(s), random_state=a.seed)
    s = pd.concat([s, rest])
s = s.sample(frac=1, random_state=a.seed).reset_index(drop=True)

rows = [{"id": r.review_id, "score": int(r.review_score), "text": r.text} for r in s.itertuples()]
os.makedirs(a.out, exist_ok=True)
test, val, train = rows[: a.n_test], rows[a.n_test : a.n_test + a.n_val], rows[a.n_test + a.n_val :]
for name, part in [("train", train), ("val", val), ("test", test)]:
    write_jsonl(f"{a.out}/{name}_raw.jsonl", part)
    print(name, len(part))
