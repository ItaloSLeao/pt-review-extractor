"""Schema, prompt e parsing compartilhados por todas as etapas."""
import json
import re

ASPECTS = ["entrega", "produto", "atendimento", "preco"]
SENTIMENTS = ["positivo", "neutro", "negativo"]
ASPECT_VALUES = ["positivo", "negativo", None]

SYSTEM_PROMPT = """Você extrai informações estruturadas de avaliações de clientes em português.
Responda APENAS com um objeto JSON válido, sem texto adicional, no formato:
{
  "sentimento": "positivo" | "neutro" | "negativo",
  "aspectos": {
    "entrega": "positivo" | "negativo" | null,
    "produto": "positivo" | "negativo" | null,
    "atendimento": "positivo" | "negativo" | null,
    "preco": "positivo" | "negativo" | null
  },
  "problema_principal": string curta (até 10 palavras) | null,
  "requer_acao": true | false
}
Regras:
- Use null para aspectos não mencionados.
- "entrega" cobre prazo, atraso, rastreio e não recebimento.
- "problema_principal" é null quando não há reclamação.
- "requer_acao" é true quando a empresa precisa agir (produto não recebido, defeito, item errado, pedido de contato)."""


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Avaliação: {text.strip()}"},
    ]
