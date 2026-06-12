# Videojuegos y Rendimiento Académico

Proyecto de Ingeniería de Datos (UFRO). Objetivo final: estimar la nota de un estudiante a partir de sus rutinas de juego, estudio y descanso (regresión, eventualmente con una red neuronal).

## Integrantes
- Cristóbal Cariqueo
- Alvaro Parra
- Jun Sáez
- Felipe Sequel

## Entrega: problema de reglas de asociación

Definimos un problema de reglas de asociación sobre el dataset del proyecto:

- **Transacción:** un estudiante; **ítem:** un par `variable=valor` (numéricas discretizadas por cuartil Q1–Q4, categóricas tal cual).
- **P1 — minería abierta:** qué patrones co-ocurren con más fuerza y qué variables resultan redundantes entre sí (insumo para la selección de features del modelo final).
- **P2 — foco en notas:** qué combinaciones de hábitos se asocian a cada banda de calificación (`grades=Q1…Q4`).
- **Algoritmo:** Apriori (`mlxtend`), con soporte, confianza y lift como métricas.

La definición formal completa está en los notebooks.

## Estructura

| Ruta | Contenido |
|---|---|
| `notebooks/miercoles10/00_analisis_completo.ipynb` | Pipeline completo de punta a punta (versión autocontenida) |
| `notebooks/miercoles10/01_exploracion_discretizacion.ipynb` | Parte 1: perfilado, limpieza de outliers, discretización por cuartiles |
| `notebooks/miercoles10/02_reglas_asociacion.ipynb` | Parte 2: definición del problema, Apriori, reglas y conclusiones |
| `notebooks/viernes12/03_export_reglas.ipynb` | Reproduce la minería del notebook 02 y exporta **todas** las reglas (sin filtro de lift) a CSV |
| `data/raw/Gaming_Academic_Performance.csv` | [Dataset original](data/raw/Gaming_Academic_Performance.csv) (Kaggle, 8000 estudiantes) — se conserva intacto como respaldo |
| `data/processed/dataset_discretizado.csv` | Dataset limpio y discretizado, generado por el notebook 01 |
| `data/processed/reglas_todas.csv` | Las 7886 reglas (todas, sin filtro de lift), generadas por el notebook 03 |

## Cómo ejecutar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab   # o jupyter notebook
```

Los notebooks ya vienen ejecutados (outputs incluidos). Para re-ejecutar desde cero: correr `01` antes que `02` y `03` (genera `data/processed/`); `00` es independiente. Los notebooks resuelven la raíz del repo automáticamente, así que el kernel puede arrancar desde la raíz o desde la carpeta del notebook.
