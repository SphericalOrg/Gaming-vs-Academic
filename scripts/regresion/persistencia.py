from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

try:
    from .config import EXCLUIR_MODELO, TARGET
except ImportError:
    from config import EXCLUIR_MODELO, TARGET  # type: ignore


def guardar_metricas(
    metricas: list[dict[str, Any]],
    predicciones: list[pd.DataFrame],
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_metricas = pd.DataFrame(metricas).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    df_predicciones = pd.concat(predicciones, ignore_index=True)

    df_metricas.to_csv(processed_dir / "modelos_regresion_metricas.csv", index=False)
    df_predicciones.to_csv(
        processed_dir / "modelos_regresion_predicciones_test.csv",
        index=False,
    )

    return df_metricas, df_predicciones


def guardar_modelo_final(
    mejor_modelo: Pipeline,
    mejor_metricas: pd.Series,
    x: pd.DataFrame,
    y: pd.Series,
    numericas: list[str],
    categoricas: list[str],
    out_dir: Path,
) -> None:
    modelo_final = clone(mejor_modelo)
    modelo_final.fit(x, y)

    modelo_path = out_dir / "mejor_modelo_regresion.joblib"
    modelo_tmp = out_dir / "mejor_modelo_regresion.joblib.tmp"
    joblib.dump(modelo_final, modelo_tmp)
    modelo_tmp.replace(modelo_path)

    metadata = {
        "modelo": str(mejor_metricas["modelo"]),
        "target": TARGET,
        "features": x.columns.tolist(),
        "features_numericas": numericas,
        "features_categoricas": categoricas,
        "features_excluidas": EXCLUIR_MODELO,
        "metricas_test": {
            "MAE": float(mejor_metricas["MAE"]),
            "RMSE": float(mejor_metricas["RMSE"]),
            "MAPE_pct": float(mejor_metricas["MAPE_pct"]),
            "R2": float(mejor_metricas["R2"]),
        },
        "nota": (
            "El modelo final se reentrena con todas las filas limpias usando la "
            "configuracion seleccionada por la evaluacion train/test."
        ),
    }
    (out_dir / "mejor_modelo_regresion_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def guardar_resumen(
    df_metricas: pd.DataFrame,
    df_limpio: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
    out_dir: Path,
) -> None:
    mejor = df_metricas.iloc[0]
    texto = f"""# Experimento de regresion

## Objetivo

Predecir `{TARGET}` en escala 0-100 usando habitos de videojuegos, estudio, descanso y uso de dispositivos.

## Preparacion de datos

- Entrada: `data/raw/Gaming_Academic_Performance.csv`.
- Filas originales: 8000.
- Se eliminaron 134 filas con `{TARGET} > 100` y 1 fila con `{TARGET} == 0`.
- Filas finales: {len(df_limpio)}.
- No se usa `data/processed/dataset_discretizado.csv` para entrenar porque ese archivo convierte `{TARGET}` a Q1-Q4 para Apriori.
- Variables excluidas del modelo: {", ".join(EXCLUIR_MODELO)}.
- Variables numericas usadas: {", ".join(numericas)}.
- Variables categoricas usadas: {", ".join(categoricas)}.

## Modelos comparados

- Promedio como referencia.
- Regresion lineal.
- Arbol de decision regresor.
- Random Forest regressor.

## Metricas en test

{tabla_markdown(df_metricas)}

## Resultado principal

El mejor modelo fue **{mejor['modelo']}**, con MAE={mejor['MAE']:.4f}, RMSE={mejor['RMSE']:.4f}, MAPE={mejor['MAPE_pct']:.4f}% y R2={mejor['R2']:.4f}.
"""
    (out_dir / "resumen_experimento_regresion.md").write_text(texto, encoding="utf-8")


def tabla_markdown(df: pd.DataFrame) -> str:
    columnas = ["modelo", "MAE", "RMSE", "MAPE_pct", "R2", "cv_mae", "mejores_parametros"]
    visible = df[columnas].copy()
    for col in ["MAE", "RMSE", "MAPE_pct", "R2", "cv_mae"]:
        visible[col] = visible[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    header = "| " + " | ".join(columnas) + " |"
    sep = "| " + " | ".join(["---"] * len(columnas)) + " |"
    filas = [
        "| " + " | ".join(str(valor) for valor in fila) + " |"
        for fila in visible.to_numpy()
    ]
    return "\n".join([header, sep, *filas])
