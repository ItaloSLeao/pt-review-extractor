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
