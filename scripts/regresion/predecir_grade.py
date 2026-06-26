from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


FEATURES = [
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "gaming_genre",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "stress_level",
]

NUMERICAS = [
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
]

CATEGORICAS = ["gaming_genre", "stress_level"]


def encontrar_repo(inicio: Path | None = None) -> Path:
    actual = (inicio or Path.cwd()).resolve()
    for ruta in [actual, *actual.parents]:
        if (ruta / "data").exists():
            return ruta
    raise FileNotFoundError("No se encontro la raiz del repo con carpeta data/.")


def cargar_entrada(args: argparse.Namespace) -> dict[str, Any]:
    if args.json:
        posible_path = Path(args.json)
        if posible_path.exists():
            return json.loads(posible_path.read_text(encoding="utf-8"))
        return json.loads(args.json)

    return {
        "gaming_hours": args.gaming_hours,
        "study_hours": args.study_hours,
        "sleep_hours": args.sleep_hours,
        "attendance": args.attendance,
        "gaming_genre": args.gaming_genre,
        "social_activity": args.social_activity,
        "device_usage": args.device_usage,
        "reaction_time_ms": args.reaction_time_ms,
        "addiction_score": args.addiction_score,
        "stress_level": args.stress_level,
    }


def validar_entrada(datos: dict[str, Any]) -> dict[str, Any]:
    faltantes = [feature for feature in FEATURES if datos.get(feature) is None]
    if faltantes:
        raise ValueError(
            "Faltan variables de entrada: "
            + ", ".join(faltantes)
            + ". Usa --help para ver un ejemplo."
        )

    limpio = {feature: datos[feature] for feature in FEATURES}
    for col in NUMERICAS:
        limpio[col] = float(limpio[col])
    for col in CATEGORICAS:
        limpio[col] = str(limpio[col])

    return limpio


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predice grades usando el mejor modelo entrenado del proyecto."
    )
    parser.add_argument(
        "--json",
        help=(
            "JSON inline o ruta a un archivo JSON con las 10 variables de entrada. "
            "Si se usa, ignora los flags individuales."
        ),
    )
    parser.add_argument("--gaming-hours", type=float)
    parser.add_argument("--study-hours", type=float)
    parser.add_argument("--sleep-hours", type=float)
    parser.add_argument("--attendance", type=float)
    parser.add_argument("--gaming-genre")
    parser.add_argument("--social-activity", type=float)
    parser.add_argument("--device-usage", type=float)
    parser.add_argument("--reaction-time-ms", type=float)
    parser.add_argument("--addiction-score", type=float)
    parser.add_argument("--stress-level")
    return parser


def main() -> None:
    args = crear_parser().parse_args()
    repo = encontrar_repo()
    model_path = repo / "artifacts" / "regresion" / "mejor_modelo_regresion.joblib"
    metadata_path = (
        repo / "artifacts" / "regresion" / "mejor_modelo_regresion_metadata.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "No existe el modelo entrenado. Primero ejecuta: "
            "python scripts/regresion/experimento_regresion.py"
        )

    datos = validar_entrada(cargar_entrada(args))
    modelo = joblib.load(model_path)
    prediccion = float(modelo.predict(pd.DataFrame([datos]))[0])
    prediccion_acotada = max(0.0, min(100.0, prediccion))

    print("Entrada usada:")
    for feature in FEATURES:
        print(f"  {feature}: {datos[feature]}")
    print()
    print(f"Grade predicho: {prediccion_acotada:.2f}")

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metricas = metadata.get("metricas_test", {})
        if metricas:
            print(
                "Referencia del modelo en test: "
                f"MAE={metricas['MAE']:.2f}, RMSE={metricas['RMSE']:.2f}, "
                f"MAPE={metricas['MAPE_pct']:.2f}%, R2={metricas['R2']:.3f}"
            )


if __name__ == "__main__":
    main()
