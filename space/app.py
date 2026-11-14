import json

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_messages, parse_output

MODEL = "SEU_USER/qwen2.5-0.5b-avaliacoes-ptbr"  # 0.5B: viável em CPU gratuita
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

EXEMPLOS = [
    "Comprei há 20 dias e até agora nada. Ninguém responde no chat.",
    "Chegou antes do prazo, produto excelente e bem embalado!",
    "O produto é bom mas veio na cor errada. Quero trocar.",
]


def extrair(texto):
    ids = tok.apply_chat_template(build_messages(texto), add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False)
    raw = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
    obj = parse_output(raw)
    return obj if obj is not None else {"erro": "saída inválida", "raw": raw}


gr.Interface(
    fn=extrair,
    inputs=gr.Textbox(lines=4, label="Avaliação do cliente"),
    outputs=gr.JSON(label="Extração estruturada"),
    examples=EXEMPLOS,
    title="Extrator de avaliações PT-BR",
    description="Qwen2.5-0.5B ajustado com QLoRA para transformar avaliações em JSON (sentimento, aspectos, problema, ação).",
).launch()
