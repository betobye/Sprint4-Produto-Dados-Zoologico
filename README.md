# Sprint 4 – Produto de Dados com Explicabilidade

Projeto desenvolvido para a Sprint 4 da disciplina de Inteligência Artificial / Machine Learning da FIAP.

O objetivo do trabalho foi transformar o modelo criado na Sprint 3 em um produto de dados funcional, permitindo prever a quantidade de visitantes de um zoológico e apresentar explicações sobre os fatores que mais influenciaram a previsão.

Sobre o projeto

A aplicação foi desenvolvida em Streamlit e utiliza um modelo de regressão treinado com dados sintéticos de visitação diária ao longo de 2023.

Além da previsão, o sistema também utiliza SHAP para explicar quais variáveis tiveram maior impacto no resultado apresentado.

Exemplos:

fim de semana
verão
sazonalidade
feriados
Tecnologias utilizadas
Python
Pandas
Scikit-learn
Streamlit
SHAP
Matplotlib
Funcionalidades

A aplicação permite:

selecionar uma data
prever a quantidade de visitantes
classificar o nível de demanda
visualizar fatores que influenciaram a previsão
Estrutura do projeto
app.py → aplicação Streamlit
train_model.py → treinamento do modelo
modelo_visitantes.pkl → modelo salvo
dados_zoologico.csv → base de dados
Sprint3_Modelagem_Avancada_Zoologico.ipynb → notebook da Sprint 3
Como executar

Instalar dependências:

pip install -r requirements.txt

Executar aplicação:

streamlit run app.py
Observações

Os dados utilizados são sintéticos e foram gerados apenas para fins acadêmicos.

O projeto tem como foco demonstrar:

pipeline de machine learning
tuning de modelos
interpretabilidade
uso prático de IA em apoio à tomada de decisão

## URL Streamlit
https://sprint4-apputo-dados-zoologico-88ypjnndzi5tomdwf9g2me.streamlit.app/





A base utilizada é sintética e foi construída para fins acadêmicos. Em uma aplicação real, seria necessário treinar o modelo com dados históricos reais de visitação, clima, eventos, feriados, campanhas, férias escolares e outras variáveis externas.
