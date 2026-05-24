# Sprint 4 – Produto de Dados com Explicabilidade

## Projeto
Previsão de visitação diária para um zoológico, transformando o modelo da Sprint 3 em um produto de dados funcional com Streamlit.

## Objetivo
A aplicação permite prever a quantidade estimada de visitantes em uma data escolhida e explicar quais variáveis influenciaram a previsão usando SHAP.

## Foco escolhido
**Previsão de visitantes por regressão.**

Esse foco foi escolhido porque a Sprint 3 já possuía um pipeline completo de regressão, com feature engineering, tuning, avaliação em holdout e interpretabilidade.

## Funcionalidades
- Entrada por data.
- Identificação de fim de semana, feriado, mês, trimestre, dia do ano e estação.
- Previsão da quantidade de visitantes.
- Classificação operacional da demanda em baixa, média ou alta.
- Explicação local com SHAP.
- Recomendação de decisão operacional para o totem.
- Avaliação crítica sobre limitações e melhorias.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Treine o modelo, caso necessário:

```bash
python train_model.py
```

Execute o aplicativo:

```bash
streamlit run app.py
```

## Arquivos

- `app.py`: aplicação Streamlit.
- `train_model.py`: script de criação da base sintética e treinamento do modelo.
- `modelo_visitantes.pkl`: modelo treinado e artefatos necessários para previsão e SHAP.
- `dados_zoologico.csv`: base sintética usada no projeto.
- `requirements.txt`: dependências do projeto.

## Conexão com o negócio

O produto ajuda o totem e a operação do zoológico a antecipar dias de maior fluxo. Com isso, é possível planejar melhor a equipe, organizar filas, preparar comunicação para visitantes e apoiar decisões operacionais.

## Limitações

A base utilizada é sintética e foi construída para fins acadêmicos. Em uma aplicação real, seria necessário treinar o modelo com dados históricos reais de visitação, clima, eventos, feriados, campanhas, férias escolares e outras variáveis externas.
