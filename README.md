# TCC – Sistemas de Recomendação em E-commerce

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso do MBA em Data Science & Analytics (USP/ESALQ).

## Descrição
O projeto implementa e avalia diferentes abordagens de sistemas de recomendação aplicadas ao contexto de e-commerce alimentar, incluindo:
- Popularidade global
- Filtragem colaborativa baseada em itens
- Recomendação baseada em conteúdo
- Modelo híbrido

A avaliação é realizada por meio de métricas Top-N e métricas de negócio, utilizando o dataset público Instacart Online Grocery Shopping Dataset.

## Dataset
Os dados utilizados são do **Instacart Online Grocery Shopping Dataset (2017)** e devem ser obtidos separadamente.

## Execução

```bash
pip install -r requirements.txt

python recsys_instacart_pipeline.py \
  --data_dir data/instacart \
  --out_dir outputs \
  --max_users 20000 \
  --K 5,10,20
