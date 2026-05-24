# ============================================================
# TREINAMENTO DO MODELO - SPRINT 4
# Produto de Dados com Explicabilidade
# ============================================================

import numpy as np
import pandas as pd
import joblib
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

FEATURES = [
    "mes", "dia", "dia_do_ano", "trimestre", "dia_semana_num",
    "is_weekend", "is_holiday", "season_enc", "is_summer",
    "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos"
]

HOLIDAYS_2023 = [
    date(2023, 1, 1), date(2023, 1, 6), date(2023, 4, 7),
    date(2023, 4, 9), date(2023, 4, 10), date(2023, 5, 1),
    date(2023, 5, 18), date(2023, 6, 23), date(2023, 6, 24),
    date(2023, 11, 4), date(2023, 12, 6), date(2023, 12, 24),
    date(2023, 12, 25), date(2023, 12, 26)
]

def get_season(mes):
    if mes in [12, 1, 2]:
        return "Winter"
    if mes in [3, 4, 5]:
        return "Spring"
    if mes in [6, 7, 8]:
        return "Summer"
    return "Autumn"

def criar_base():
    datas = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    registros = []

    for dt in datas:
        mes = dt.month
        dia_semana_num = dt.dayofweek
        is_weekend = 1 if dia_semana_num >= 5 else 0
        is_holiday = 1 if dt.date() in HOLIDAYS_2023 else 0

        base_sazonal = 1800 + 2200 * np.sin(np.pi * (mes - 1) / 11)
        fator_fds = 1.30 if is_weekend else 1.00
        fator_feriado = 1.15 if is_holiday else 1.00
        ruido = np.random.uniform(0.80, 1.20)

        visitors = int(base_sazonal * fator_fds * fator_feriado * ruido)
        visitors = max(visitors, 50)

        registros.append({
            "date": dt,
            "mes": mes,
            "dia": dt.day,
            "dia_semana_num": dia_semana_num,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "visitors": visitors
        })

    df = pd.DataFrame(registros)
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}

    df["dia_do_ano"] = df["date"].dt.dayofyear
    df["trimestre"] = df["date"].dt.quarter
    df["season"] = df["mes"].apply(get_season)
    df["season_enc"] = df["season"].map(season_map)
    df["is_summer"] = df["mes"].isin([6, 7, 8]).astype(int)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana_num"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana_num"] / 7)

    return df

if __name__ == "__main__":
    df = criar_base()

    X = df[FEATURES]
    y = df["visitors"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo escolhido para o produto: Gradient Boosting Regressor.
    # Ele é adequado para dados tabulares e compatível com explicabilidade via SHAP.
    modelo = GradientBoostingRegressor(
        random_state=42,
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        min_samples_split=4
    )

    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    metricas = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred))
    }

    pacote = {
        "modelo": modelo,
        "features": FEATURES,
        "background": X_train,
        "metricas": metricas
    }

    df.to_csv("dados_zoologico.csv", index=False)
    joblib.dump(pacote, "modelo_visitantes.pkl")

    print("Modelo treinado e salvo com sucesso.")
    print(metricas)
