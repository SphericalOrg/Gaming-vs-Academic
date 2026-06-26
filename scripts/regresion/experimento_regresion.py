from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .config import EXCLUIR_MODELO, RANDOM_STATE, TARGET, TEST_SIZE
    from .datos import cargar_y_limpiar, encontrar_repo, separar_xy
    from .graficos import guardar_graficos, guardar_importancias
    from .modelos import ajustar_modelo, metricas_regresion, preparar_modelos
    from .persistencia import (
        guardar_metricas,
        guardar_modelo_final,
        guardar_resumen,
    )
except ImportError:
    from config import EXCLUIR_MODELO, RANDOM_STATE, TARGET, TEST_SIZE  # type: ignore
    from datos import cargar_y_limpiar, encontrar_repo, separar_xy  # type: ignore
    from graficos import guardar_graficos, guardar_importancias  # type: ignore
    from modelos import ajustar_modelo, metricas_regresion, preparar_modelos  # type: ignore
    from persistencia import (  # type: ignore
        guardar_metricas,
        guardar_modelo_final,
        guardar_resumen,
    )


def evaluar_modelos(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numericas: list[str],
    categoricas: list[str],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Pipeline]]:
    metricas = []
    predicciones = []
    modelos_ajustados = {}

    for spec in preparar_modelos(numericas, categoricas):
        nombre = spec["nombre"]
        modelo, mejores_params, cv_mae = ajustar_modelo(
            pipeline=spec["pipeline"],
            param_grid=spec["param_grid"],
            x_train=x_train,
            y_train=y_train,
        )
        y_pred = modelo.predict(x_test)

        metricas.append(
            {
                "modelo": nombre,
                **metricas_regresion(y_test, y_pred),
                "cv_mae": cv_mae,
                "mejores_parametros": mejores_params,
            }
        )
        modelos_ajustados[nombre] = modelo
        predicciones.append(_predicciones_test(nombre, y_test, y_pred))

    return metricas, predicciones, modelos_ajustados


def _predicciones_test(
    nombre_modelo: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    error = y_test.to_numpy() - y_pred
    return pd.DataFrame(
        {
            "modelo": nombre_modelo,
            "fila_limpia": y_test.index,
            "grades_real": y_test.to_numpy(),
            "grades_predicho": y_pred,
            "error": error,
            "error_abs": np.abs(error),
            "ape_pct": np.abs(error / y_test.to_numpy()) * 100,
        }
    )


def main() -> None:
    repo = encontrar_repo()
    raw_csv = repo / "data" / "raw" / "Gaming_Academic_Performance.csv"
    processed_dir = repo / "data" / "processed"
    out_dir = repo / "artifacts" / "regresion"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_limpio = cargar_y_limpiar(raw_csv)
    df_limpio.to_csv(processed_dir / "dataset_regresion_limpio.csv", index=False)

    x, y, numericas, categoricas = separar_xy(df_limpio)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metricas, predicciones, modelos_ajustados = evaluar_modelos(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        numericas=numericas,
        categoricas=categoricas,
    )
    df_metricas, df_predicciones = guardar_metricas(
        metricas=metricas,
        predicciones=predicciones,
        processed_dir=processed_dir,
    )

    mejor_nombre = str(df_metricas.iloc[0]["modelo"])
    mejor_modelo = modelos_ajustados[mejor_nombre]
    mejor_metricas = df_metricas.iloc[0]

    guardar_graficos(df_metricas, df_predicciones, mejor_nombre, out_dir)
    guardar_importancias(mejor_modelo, mejor_nombre, out_dir)
    guardar_modelo_final(
        mejor_modelo=mejor_modelo,
        mejor_metricas=mejor_metricas,
        x=x,
        y=y,
        numericas=numericas,
        categoricas=categoricas,
        out_dir=out_dir,
    )
    guardar_resumen(df_metricas, df_limpio, numericas, categoricas, out_dir)

    print("Experimento de regresion completado.")
    print(f"Filas limpias: {len(df_limpio)}")
    print(f"Mejor modelo: {mejor_nombre}")
    print(df_metricas[["modelo", "MAE", "RMSE", "MAPE_pct", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
