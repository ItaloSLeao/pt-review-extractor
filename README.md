# Extrator estruturado de avaliações PT-BR

Fine-tuning QLoRA de Qwen2.5 (0.5B e 1.5B) para converter avaliações de clientes em português em JSON:
sentimento, sentimento por aspecto (entrega, produto, atendimento, preço), problema principal e se requer ação.

Pipeline: destilação de um professor (Qwen2.5-7B-Instruct) → revisão manual do conjunto de teste → QLoRA → avaliação contra baselines → publicação no HF.

## Limitações
- Dados de e-commerce (Olist), não de delivery de comida.
- Rótulos de treino vêm de um modelo professor (destilação); só o teste foi revisado manualmente.
