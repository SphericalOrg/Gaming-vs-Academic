# Experimento de regresion

## Objetivo

Predecir `grades` en escala 0-100 usando habitos de videojuegos, estudio, descanso y uso de dispositivos.

## Preparacion de datos

- Entrada: `data/raw/Gaming_Academic_Performance.csv`.
- Filas originales: 8000.
- Se eliminaron 134 filas con `grades > 100` y 1 fila con `grades == 0`.
- Filas finales: 7865.
- No se usa `data/processed/dataset_discretizado.csv` para entrenar porque ese archivo convierte `grades` a Q1-Q4 para Apriori.
- Variables excluidas del modelo: student_id, age, gender.
- Variables numericas usadas: gaming_hours, study_hours, sleep_hours, attendance, social_activity, device_usage, reaction_time_ms, addiction_score.
- Variables categoricas usadas: gaming_genre, stress_level.

## Modelos comparados

- Promedio como referencia.
- Regresion lineal.
- Arbol de decision regresor.
- Random Forest regressor.

## Metricas en test

| modelo | MAE | RMSE | MAPE_pct | R2 | cv_mae | mejores_parametros |
| --- | --- | --- | --- | --- | --- | --- |
| Random forest | 4.7673 | 6.1157 | 9.5604 | 0.9211 | 4.6492 | {'modelo__max_depth': None, 'modelo__min_samples_leaf': 5} |
| Regresion lineal | 5.2531 | 6.6252 | 10.4449 | 0.9074 |  | {} |
| Arbol de decision | 5.6565 | 7.2641 | 11.3743 | 0.8887 | 5.6349 | {'modelo__max_depth': 12, 'modelo__min_samples_leaf': 15} |
| Promedio (referencia) | 17.9820 | 21.7779 | 43.6347 | -0.0004 |  | {} |

## Resultado principal

El mejor modelo fue **Random forest**, con MAE=4.7673, RMSE=6.1157, MAPE=9.5604% y R2=0.9211.
