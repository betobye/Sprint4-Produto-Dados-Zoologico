# ============================================================
# SPRINT 4 - PRODUTO DE DADOS COM EXPLICABILIDADE
# Aplicação Streamlit para previsão de visitantes do zoológico
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(
    page_title="Previsão de Visitantes - Zoológico",
    page_icon="🦁",
    layout="wide"
)

FEATURES = [
    "mes", "dia", "dia_do_ano", "trimestre", "dia_semana_num",
    "is_weekend", "is_holiday", "season_enc", "is_summer",
    "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos"
]

FEATURE_LABELS = {
    "mes": "Mês",
    "dia": "Dia do mês",
    "dia_do_ano": "Dia do ano",
    "trimestre": "Trimestre",
    "dia_semana_num": "Dia da semana",
    "is_weekend": "Final de semana",
    "is_holiday": "Feriado",
    "season_enc": "Estação do ano",
    "is_summer": "Verão",
    "mes_sin": "Sazonalidade do mês",
    "mes_cos": "Ciclo do mês",
    "dia_semana_sin": "Sazonalidade semanal",
    "dia_semana_cos": "Ciclo semanal"
}

HOLIDAYS_2023 = [
    date(2023, 1, 1), date(2023, 1, 6), date(2023, 4, 7),
    date(2023, 4, 9), date(2023, 4, 10), date(2023, 5, 1),
    date(2023, 5, 18), date(2023, 6, 23), date(2023, 6, 24),
    date(2023, 11, 4), date(2023, 12, 6), date(2023, 12, 24),
    date(2023, 12, 25), date(2023, 12, 26)
]

SEASON_MAP = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}

def get_season(mes):
    if mes in [12, 1, 2]:
        return "Winter"
    if mes in [3, 4, 5]:
        return "Spring"
    if mes in [6, 7, 8]:
        return "Summer"
    return "Autumn"

def traduzir_estacao(season):
    return {
        "Winter": "Inverno",
        "Spring": "Primavera",
        "Summer": "Verão",
        "Autumn": "Outono"
    }.get(season, season)

def criar_features(data_escolhida, forcar_feriado=None):
    dt = pd.to_datetime(data_escolhida)
    mes = dt.month
    dia_semana_num = dt.dayofweek
    season = get_season(mes)

    is_weekend = 1 if dia_semana_num >= 5 else 0
    is_holiday = 1 if dt.date() in HOLIDAYS_2023 else 0

    if forcar_feriado is not None:
        is_holiday = 1 if forcar_feriado else 0

    registro = {
        "mes": mes,
        "dia": dt.day,
        "dia_do_ano": dt.dayofyear,
        "trimestre": dt.quarter,
        "dia_semana_num": dia_semana_num,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "season_enc": SEASON_MAP[season],
        "is_summer": 1 if mes in [6, 7, 8] else 0,
        "mes_sin": np.sin(2 * np.pi * mes / 12),
        "mes_cos": np.cos(2 * np.pi * mes / 12),
        "dia_semana_sin": np.sin(2 * np.pi * dia_semana_num / 7),
        "dia_semana_cos": np.cos(2 * np.pi * dia_semana_num / 7)
    }

    return pd.DataFrame([registro])[FEATURES], season

def classificar_demanda(previsao):
    if previsao < 2600:
        return "Baixa", "Operação normal, sem necessidade de reforço especial."
    if previsao < 4200:
        return "Média", "Monitorar fluxo e preparar equipe de apoio nos horários de pico."
    return "Alta", "Reforçar equipe, orientar filas e preparar comunicação ativa no totem."

def gerar_explicacao_negocio(linha, season, demanda):
    motivos = []

    if int(linha["is_weekend"].iloc[0]) == 1:
        motivos.append("fim de semana")
    if int(linha["is_holiday"].iloc[0]) == 1:
        motivos.append("feriado")
    if season == "Summer":
        motivos.append("verão")
    if int(linha["mes"].iloc[0]) in [6, 7, 8]:
        motivos.append("período sazonal favorável")

    if motivos:
        return f"A demanda {demanda.lower()} está associada principalmente a " + ", ".join(motivos) + "."

    return f"A demanda {demanda.lower()} está associada a um dia comum, sem grandes fatores de aumento de visitação."

@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_visitantes.pkl")

pacote = carregar_modelo()
modelo = pacote["modelo"]
background = pacote["background"]
metricas = pacote["metricas"]

st.title("Produto de Dados: Previsão de Visitantes do Zoológico")
st.write(
    "Aplicação da Sprint 4 para transformar o modelo da Sprint 3 em uma solução funcional, "
    "com previsão, explicabilidade e suporte à decisão operacional."
)

with st.sidebar:
    st.header("Entrada de dados")
    data_escolhida = st.date_input(
        "Escolha a data para previsão",
        value=date(2023, 7, 15),
        min_value=date(2023, 1, 1),
        max_value=date(2023, 12, 31)
    )

    usar_feriado_manual = st.checkbox("Informar feriado manualmente")
    feriado_manual = None
    if usar_feriado_manual:
        feriado_manual = st.checkbox("Esta data é feriado?")

    st.divider()
    st.caption("A base sintética foi construída para o ano de 2023, seguindo a Sprint 3.")

linha_predicao, season = criar_features(data_escolhida, feriado_manual)
previsao = float(modelo.predict(linha_predicao)[0])
demanda, acao = classificar_demanda(previsao)
explicacao_negocio = gerar_explicacao_negocio(linha_predicao, season, demanda)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Visitantes previstos", f"{previsao:,.0f}".replace(",", "."))

with col2:
    st.metric("Nível de demanda", demanda)

with col3:
    st.metric("Estação", traduzir_estacao(season))

st.subheader("Recomendação para operação")
st.info(acao)
st.write(explicacao_negocio)

st.subheader("Explicabilidade local com SHAP")
st.write(
    "O gráfico abaixo mostra quais variáveis mais aumentaram ou reduziram a previsão para a data escolhida."
)

explainer = shap.Explainer(modelo, background)
shap_values = explainer(linha_predicao)

contribuicoes = pd.DataFrame({
    "feature": FEATURES,
    "feature_traduzida": [FEATURE_LABELS[f] for f in FEATURES],
    "valor": linha_predicao.iloc[0].values,
    "impacto_shap": shap_values.values[0]
})

contribuicoes["impacto_abs"] = contribuicoes["impacto_shap"].abs()
top_contribuicoes = contribuicoes.sort_values("impacto_abs", ascending=False).head(6)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top_contribuicoes["feature_traduzida"], top_contribuicoes["impacto_shap"])
ax.set_xlabel("Impacto na previsão de visitantes")
ax.set_ylabel("Feature")
ax.set_title("Principais fatores da previsão")
ax.invert_yaxis()
st.pyplot(fig)

st.dataframe(
    top_contribuicoes[["feature_traduzida", "valor", "impacto_shap"]].rename(columns={
        "feature_traduzida": "Feature",
        "valor": "Valor usado",
        "impacto_shap": "Impacto SHAP"
    }),
    use_container_width=True
)

st.subheader("Qual o benefício para o totem?")
st.write(
    "Com a previsão de visitação, o totem pode apoiar a operação indicando dias de maior demanda, "
    "antecipando filas, necessidade de reforço na equipe, melhor organização de atendimento e comunicação "
    "mais direcionada ao público."
)

st.subheader("Avaliação crítica")
st.warning(
    "Limitação: a base usada é sintética e representa o ano de 2023. Para uso real, o modelo deve ser treinado "
    "com dados históricos reais do zoológico, clima, eventos, campanhas, férias escolares e bilheteria."
)

with st.expander("Métricas do modelo"):
    st.write(f"MAE: {metricas['MAE']:.2f} visitantes")
    st.write(f"RMSE: {metricas['RMSE']:.2f} visitantes")
    st.write(f"R²: {metricas['R2']:.4f}")
