# Extrator estruturado de avaliações PT-BR

[![Modelo](https://img.shields.io/badge/Hugging%20Face-modelo-yellow)](https://huggingface.co/aitalo/qwen2.5-0.5b-avaliacoes-ptbr)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-dataset-yellow)](https://huggingface.co/datasets/aitalo/avaliacoes-ptbr-estruturadas)
[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItaloSLeao/pt-review-extractor/blob/main/notebooks/demo_colab.ipynb)

Fine-tuning com QLoRA do Qwen2.5-0.5B-Instruct para transformar avaliações de clientes de e-commerce em português em JSON estruturado: sentimento geral, sentimento por aspecto (entrega, produto, atendimento, preço), problema principal e se a empresa precisa agir.

Entrada:

```
Comprei 2 unidades e só veio 1. Ninguém responde no SAC.
```

Saída:

```json
{
  "sentimento": "negativo",
  "aspectos": {"entrega": "negativo", "produto": null, "atendimento": "negativo", "preco": null},
  "problema_principal": "falta de produto",
  "requer_acao": true
}
```

## Resultados

Conjunto de teste de 297 avaliações com rótulos revisados pelo autor. Saídas que não são JSON válido contam como erro em todas as métricas.

| Métrica | Base 0.5B | Ajustado |
|---|---|---|
| JSON válido | 0,185 | **0,956** |
| F1 sentimento (macro) | 0,212 | **0,760** |
| F1 entrega | 0,104 | **0,893** |
| F1 produto | 0,189 | **0,844** |
| F1 atendimento | 0,035 | **0,733** |
| F1 preço | 0,046 | **0,757** |
| F1 requer ação | 0,219 | **0,823** |

Uma única rodada de treino, sem repetição com outras sementes. A classe `neutro` tem ~23 exemplos no teste, e `atendimento` e `preço` aparecem pouco, então o F1 nelas é instável.

## Pipeline

1. `src/prepare.py`: amostra balanceada por nota de avaliações da Olist e separação em treino, validação e teste.
2. `src/label.py`: rotula com um modelo professor (Qwen2.5-7B-Instruct-AWQ, via vLLM), validando cada saída contra o esquema JSON. Processa em blocos e retoma de onde parou.
3. Revisão manual do conjunto de teste (297 avaliações), com sugestões de correção geradas por um LLM e conferidas pelo autor.
4. `src/train.py`: QLoRA 4-bit (NF4), r=16, 2 épocas, lote efetivo 16, em uma GPU T4 (~25 min).
5. `src/evaluate.py`: F1 por campo, taxa de JSON válido e tempo por amostra, comparando o modelo base com o ajustado.
6. `src/push.py`: publica dataset e modelo (fundido, em fp16) no Hugging Face Hub.

O esquema e o prompt ficam em `src/common.py`.

## Como reproduzir (Kaggle, GPU T4 x2, internet ligada)

```bash
pip install -q -U "trl>=0.20" "peft>=0.13" bitsandbytes datasets accelerate
python src/prepare.py --csv <caminho>/olist_order_reviews_dataset.csv

# em outro notebook, para não conflitar dependências:
pip install -q vllm && python src/label.py

# revisar data/test_review.csv e salvar como data/test_gold.csv

CUDA_VISIBLE_DEVICES=0 python src/train.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen0.5b

pip uninstall -y torchao   # evita conflito de versão com o PEFT no Kaggle
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --name base-0.5b
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen0.5b --name ft-0.5b
```

## Demonstração

O notebook `notebooks/demo_colab.ipynb` carrega o modelo do Hub e abre uma interface interativa. Use o botão "Abrir no Colab" acima, selecione GPU T4 e execute tudo.

## Limitações

- Domínio: avaliações de e-commerce da Olist, não de delivery de comida.
- Os rótulos de treino vêm de um modelo professor e carregam os erros dele; só o teste foi revisado.
- 4,4% das saídas no teste não foram JSON válido.
- A amostragem balanceada por nota faz o teste ter mais avaliações negativas do que a distribuição real.

## Dados e licença

Os textos vêm do *Brazilian E-Commerce Public Dataset by Olist* (Kaggle), sob CC BY-NC-SA 4.0 (uso não comercial). O dataset derivado mantém essa licença. O modelo é um projeto de estudo e portfólio.
