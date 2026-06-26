from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import (
        CATEGORICAS_ESPERADAS,
        COLUMNAS_ESPERADAS,
        EXCLUIR_MODELO,
        NUMERICAS_ESPERADAS,
        TARGET,
    )
except ImportError:
    from config import (  # type: ignore
        CATEGORICAS_ESPERADAS,
        COLUMNAS_ESPERADAS,
        EXCLUIR_MODELO,
        NUMERICAS_ESPERADAS,
        TARGET,
    )


def encontrar_repo(inicio: Path | None = None) -> Path:
    actual = (inicio or Path.cwd()).resolve()
    for ruta in [actual, *actual.parents]:
        if (ruta / "data").exists():
            return ruta
    raise FileNotFoundError("No se encontró la raíz del repo con carpeta data/.")


def cargar_y_limpiar(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)

    if list(df.columns) != COLUMNAS_ESPERADAS:
        raise ValueError(
            "Las columnas del dataset no coinciden con la estructura esperada."
        )

    n_sobre_100 = int((df[TARGET] > 100).sum())
    n_cero = int((df[TARGET] == 0).sum())
    df_limpio = df[(df[TARGET] <= 100) & (df[TARGET] > 0)].copy()

    assert len(df) == 8000, f"Filas raw esperadas: 8000, obtenido: {len(df)}"
    assert n_sobre_100 == 134, f"Notas > 100 esperadas: 134, obtenido: {n_sobre_100}"
    assert n_cero == 1, f"Notas == 0 esperadas: 1, obtenido: {n_cero}"
    assert len(df_limpio) == 7865, (
        f"Filas limpias esperadas: 7865, obtenido: {len(df_limpio)}"
    )
    assert df_limpio[TARGET].between(0, 100).all()

    return df_limpio.reset_index(drop=True)


def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    columnas_excluidas = EXCLUIR_MODELO + [TARGET]
    x = df.drop(columns=columnas_excluidas)
    y = df[TARGET]

    numericas = x.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = x.select_dtypes(exclude=[np.number]).columns.tolist()

    assert numericas == NUMERICAS_ESPERADAS
    assert categoricas == CATEGORICAS_ESPERADAS

    return x, y, numericas, categoricas
