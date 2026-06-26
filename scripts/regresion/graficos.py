from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline


def guardar_importancias(
    mejor_modelo: Pipeline,
    nombre_modelo: str,
    out_dir: Path,
) -> pd.DataFrame | None:
    modelo = mejor_modelo.named_steps["modelo"]
    if not hasattr(modelo, "feature_importances_"):
        return None

    pre = mejor_modelo.named_steps["preprocesamiento"]
    importancias = pd.DataFrame(
        {
            "variable": pre.get_feature_names_out(),
            "importancia": modelo.feature_importances_,
        }
    ).sort_values("importancia", ascending=False)

    importancias.to_csv(out_dir / "importancia_variables_mejor_modelo.csv", index=False)

    top = importancias.head(12).sort_values("importancia")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top, x="importancia", y="variable", color="#2f6f9f")
    plt.title(f"Importancia de variables - {nombre_modelo}")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(out_dir / "importancia_variables_mejor_modelo.png", dpi=180)
    plt.close()

    return importancias


def guardar_graficos(
    df_metricas: pd.DataFrame,
    df_predicciones: pd.DataFrame,
    mejor_nombre: str,
    out_dir: Path,
) -> None:
    _guardar_metricas(df_metricas, out_dir)

    pred_mejor = df_predicciones[df_predicciones["modelo"] == mejor_nombre].copy()
    _guardar_predicho_vs_real(pred_mejor, mejor_nombre, out_dir)
    _guardar_residuos(pred_mejor, mejor_nombre, out_dir)


def _guardar_metricas(df_metricas: pd.DataFrame, out_dir: Path) -> None:
    metricas_largas = df_metricas.melt(
        id_vars="modelo",
        value_vars=["MAE", "RMSE", "MAPE_pct"],
        var_name="metrica",
        value_name="valor",
    )

    g = sns.catplot(
        data=metricas_largas,
        kind="bar",
        x="modelo",
        y="valor",
        col="metrica",
        sharey=False,
        color="#2f6f9f",
        height=4,
        aspect=1,
    )
    g.set_xticklabels(rotation=25, ha="right")
    g.set_axis_labels("", "Valor")
    g.set_titles("{col_name}")
    g.figure.subplots_adjust(top=0.82)
    g.figure.suptitle("Comparacion de modelos de regresion")
    g.figure.savefig(
        out_dir / "metricas_modelos_regresion.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(g.figure)


def _guardar_predicho_vs_real(
    pred_mejor: pd.DataFrame,
    mejor_nombre: str,
    out_dir: Path,
) -> None:
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=pred_mejor,
        x="grades_real",
        y="grades_predicho",
        alpha=0.45,
        edgecolor=None,
        color="#2f6f9f",
    )
    minimo = min(pred_mejor["grades_real"].min(), pred_mejor["grades_predicho"].min())
    maximo = max(pred_mejor["grades_real"].max(), pred_mejor["grades_predicho"].max())
    plt.plot([minimo, maximo], [minimo, maximo], color="#c0392b", linewidth=2)
    plt.title(f"Predicho vs real - {mejor_nombre}")
    plt.xlabel("Nota real")
    plt.ylabel("Nota predicha")
    plt.tight_layout()
    plt.savefig(out_dir / "predicho_vs_real_mejor_modelo.png", dpi=180)
    plt.close()


def _guardar_residuos(
    pred_mejor: pd.DataFrame,
    mejor_nombre: str,
    out_dir: Path,
) -> None:
    pred_mejor["residuo"] = pred_mejor["grades_real"] - pred_mejor["grades_predicho"]
    plt.figure(figsize=(8, 5))
    sns.histplot(pred_mejor["residuo"], kde=True, bins=35, color="#2f6f9f")
    plt.axvline(0, color="#c0392b", linewidth=2)
    plt.title(f"Distribucion de residuos - {mejor_nombre}")
    plt.xlabel("Real - predicho")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_dir / "residuos_mejor_modelo.png", dpi=180)
    plt.close()
