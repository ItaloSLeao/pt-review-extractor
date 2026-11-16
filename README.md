# Extrator de avaliações em português

Modelo para transformar avaliações de clientes em JSON com sentimento geral, sentimento por aspecto, problema principal e necessidade de ação. Ajustei o Qwen2.5-0.5B-Instruct com QLoRA usando avaliações de e-commerce da Olist, rotuladas por um modelo professor.

O [modelo ajustado](https://huggingface.co/aitalo/qwen2.5-0.5b-avaliacoes-ptbr) e o [dataset estruturado](https://huggingface.co/datasets/aitalo/avaliacoes-ptbr-estruturadas) estão no Hugging Face. O projeto foi desenvolvido para estudo e portfólio.

## Como usar

Abra a [demonstração no Colab](https://colab.research.google.com/github/ItaloSLeao/pt-review-extractor/blob/main/notebooks/demo_colab.ipynb), selecione GPU T4 em **Ambiente de execução → Alterar tipo de ambiente** e execute as células em ordem. O notebook baixa o modelo e abre uma interface para digitar avaliações. Também funciona em CPU, com maior tempo de resposta.

Uma entrada como:

```text
Comprei 2 unidades e só veio 1. Ninguém responde no SAC.
```

tem o seguinte formato de saída. Este exemplo é ilustrativo; a resposta pode variar:

```json
{
  "sentimento": "negativo",
  "aspectos": {
    "entrega": "negativo",
    "produto": null,
    "atendimento": "negativo",
    "preco": null
  },
  "problema_principal": "falta de produto",
  "requer_acao": true
}
```

O sentimento geral pode ser `positivo`, `neutro` ou `negativo`. Cada aspecto aceita `positivo`, `negativo` ou `null` quando não é mencionado. `problema_principal` contém uma descrição curta ou `null`, e `requer_acao` indica se a empresa precisa agir.

O prompt e a validação ficam em [src/common.py](src/common.py). Quando a geração não passa pela validação, a demonstração retorna um erro com a saída bruta; quem integrar o modelo precisa tratar esse caso.

## Como reproduzir

O treino foi feito no Kaggle, em uma GPU T4. Para reproduzir, habilite GPU e internet e disponibilize o arquivo `olist_order_reviews_dataset.csv` do dataset da Olist. Execute os comandos na raiz do repositório.

### 1. Preparar os dados

Instale as dependências e substitua o caminho do CSV:

```bash
pip install -r requirements.txt
python src/prepare.py --csv /caminho/olist_order_reviews_dataset.csv
```

O script remove textos duplicados, seleciona uma amostra balanceada por nota e separa treino, validação e teste. Por padrão, solicita 5000, 300 e 300 avaliações, respectivamente, com seed 42. Os arquivos são gravados em `data/`.

### 2. Rotular e revisar o teste

Use outro notebook ou ambiente para a rotulagem, pois o vLLM pode conflitar com as dependências de treino. Disponibilize nele os arquivos da etapa anterior e execute:

```bash
pip install pandas vllm
python src/label.py
```

O professor é o Qwen2.5-7B-Instruct-AWQ. O script valida as respostas, descarta as inválidas e salva o progresso em blocos para permitir a retomada. Ele gera os arquivos `data/*_labeled.jsonl` e `data/test_review.csv`.

Revise os rótulos de `test_review.csv` e salve o resultado como `data/test_gold.csv`, mantendo as colunas. Nos aspectos sem menção, deixe o campo vazio; em `requer_acao`, use `true` ou `false`. Na rodada deste projeto, revisei 297 avaliações, com sugestões de correção de um LLM conferidas por mim.

### 3. Treinar e avaliar

No ambiente de treino, disponibilize os arquivos rotulados e o teste revisado:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen0.5b
```

O ajuste usa QLoRA em 4 bits (NF4), rank 16, duas épocas e lote efetivo de 16 exemplos. A rodada levou cerca de 25 minutos em uma T4. O adaptador e o histórico de treino ficam em `runs/qwen0.5b/`.

Compare o modelo base com o ajustado no mesmo teste:

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --name base-0.5b
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen0.5b --name ft-0.5b
```

Se houver conflito entre `torchao` e PEFT no Kaggle, remova `torchao` com `pip uninstall -y torchao` antes da avaliação. Cada execução grava métricas em `results/<nome>.json` e as gerações em `results/<nome>_outputs.csv`.

O script [src/push.py](src/push.py) permite publicar o dataset e o modelo com o adaptador fundido em fp16 no Hugging Face. A publicação exige autenticação com permissão de escrita.

## Resultados

Avaliação em 297 textos com rótulos revisados por mim. Os dois modelos usam o mesmo prompt e geração sem amostragem. Saídas que não passam pelo parser e pela validação do esquema contam como erro nos campos avaliados.

| Métrica | Base 0.5B | Ajustado |
|---|---|---|
| JSON válido | 0,185 | 0,956 |
| F1 sentimento (macro) | 0,212 | 0,760 |
| F1 entrega | 0,104 | 0,893 |
| F1 produto | 0,189 | 0,844 |
| F1 atendimento | 0,035 | 0,733 |
| F1 preço | 0,046 | 0,757 |
| F1 requer ação | 0,219 | 0,823 |

Os F1 são médias macro por campo, na escala de 0 a 1. A taxa de JSON válido mede também a conformidade com o esquema; ela não garante que a interpretação da avaliação esteja correta. O campo `problema_principal` não tem uma métrica de qualidade nesta avaliação.

Foi feita uma única rodada de treino, com seed 42. A classe `neutro` tem cerca de 23 exemplos no teste, e os aspectos `atendimento` e `preco` aparecem pouco, o que torna as métricas mais sensíveis a poucos erros. Os comandos acima geram novos arquivos de resultados; a tabela deste README não é atualizada automaticamente.

## Limitações

O modelo foi treinado e avaliado com textos de e-commerce da Olist. O desempenho em restaurantes, delivery e outros domínios não foi medido.

Os rótulos de treino e validação vêm do modelo professor e podem reproduzir os erros dele. Apenas o conjunto de teste foi revisado. A amostragem balanceada por nota também faz o teste conter mais avaliações negativas do que a distribuição original.

No teste, 4,4% das saídas não passaram pela validação. Mesmo uma saída válida pode atribuir sentimento, aspecto ou necessidade de ação incorretamente. A avaliação usa uma única rodada, sem medir a variação entre sementes.

## Dados e licença

Os textos vêm do *Brazilian E-Commerce Public Dataset by Olist*, disponibilizado no Kaggle sob CC BY-NC-SA 4.0. O dataset derivado mantém essa licença, com uso não comercial. Considere essa restrição ao reutilizar os dados e o projeto.
