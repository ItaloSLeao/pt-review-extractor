# Extrator estruturado de avaliações PT-BR

Fine-tuning QLoRA de Qwen2.5 (0.5B e 1.5B) para converter avaliações de clientes em português em JSON:
sentimento, sentimento por aspecto (entrega, produto, atendimento, preço), problema principal e se requer ação.

Pipeline: destilação de um professor (Qwen2.5-7B-Instruct) → revisão manual do conjunto de teste → QLoRA → avaliação contra baselines → publicação no HF.

## Execução (Kaggle, GPU T4 x2, internet ligada, dataset `olistbr/brazilian-ecommerce` anexado)

```bash
pip install -q -U "trl>=0.20" "peft>=0.13" bitsandbytes datasets accelerate
python src/prepare.py --csv /kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv

# notebook separado para não conflitar dependências:
pip install -q vllm && python src/label.py

# revisar data/test_review.csv → salvar data/test_gold.csv

python src/train.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen0.5b
python src/train.py --model Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen1.5b

python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --name base-0.5b
python src/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --name base-1.5b
python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen0.5b --name ft-0.5b
python src/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/qwen1.5b --name ft-1.5b
```

Referência superior: avaliar o próprio professor (7B) no teste revisado mostra quanto o aluno recupera.

## Resultados

| Modelo | JSON válido | Sentimento F1 | Aspectos F1 (média) | Requer ação F1 | ms/amostra |
|---|---|---|---|---|---|
| Qwen2.5-0.5B base | | | | | |
| Qwen2.5-0.5B + QLoRA | | | | | |
| Qwen2.5-1.5B base | | | | | |
| Qwen2.5-1.5B + QLoRA | | | | | |

## Limitações
- Dados de e-commerce (Olist), não de delivery de comida.
- Rótulos de treino vêm de um modelo professor (destilação); só o teste foi revisado manualmente.
