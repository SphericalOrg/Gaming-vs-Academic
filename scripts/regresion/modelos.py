from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from .config import CV_FOLDS, RANDOM_STATE
except ImportError:
    from config import CV_FOLDS, RANDOM_STATE  # type: ignore


def preprocesador(
    numericas: list[str],
    categoricas: list[str],
    escalar: bool,
) -> ColumnTransformer:
    pasos_numericos: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if escalar:
        pasos_numericos.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(pasos_numericos)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numericas),
            ("cat", categorical_pipe, categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preparar_modelos(numericas: list[str], categoricas: list[str]) -> list[dict[str, Any]]:
    pre_lineal = preprocesador(numericas, categoricas, escalar=True)
    pre_arbol = preprocesador(numericas, categoricas, escalar=False)

    return [
        {
            "nombre": "Promedio (referencia)",
            "pipeline": Pipeline(
                [
                    ("preprocesamiento", pre_arbol),
                    ("modelo", DummyRegressor(strategy="mean")),
                ]
            ),
            "param_grid": None,
        },
        {
            "nombre": "Regresion lineal",
            "pipeline": Pipeline(
                [
                    ("preprocesamiento", pre_lineal),
                    ("modelo", LinearRegression()),
                ]
            ),
            "param_grid": None,
        },
        {
            "nombre": "Arbol de decision",
            "pipeline": Pipeline(
                [
                    ("preprocesamiento", pre_arbol),
                    ("modelo", DecisionTreeRegressor(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "modelo__max_depth": [3, 5, 8, 12, None],
                "modelo__min_samples_leaf": [1, 5, 15, 30],
            },
        },
        {
            "nombre": "Random forest",
            "pipeline": Pipeline(
                [
                    ("preprocesamiento", pre_arbol),
                    (
                        "modelo",
                        RandomForestRegressor(
                            n_estimators=200,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "modelo__max_depth": [8, 12, None],
                "modelo__min_samples_leaf": [1, 5, 15],
            },
        },
    ]


def ajustar_modelo(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]] | None,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    nombre: str | None = None,
) -> tuple[Pipeline, dict[str, Any], float | None]:
    if not param_grid:
        pipeline.fit(x_train, y_train)
        return pipeline, {}, None

    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=1,
        refit=True,
    )
    busqueda.fit(x_train, y_train)
    return busqueda.best_estimator_, busqueda.best_params_, float(-busqueda.best_score_)


def metricas_regresion(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAPE_pct": float(mean_absolute_percentage_error(y_real, y_pred) * 100),
        "R2": float(r2_score(y_real, y_pred)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
    }
