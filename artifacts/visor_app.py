"""Visor interactivo de reglas de asociacion (Gaming vs Academic).

Dashboard Streamlit para explorar las 7886 reglas Apriori: filtra por metricas
(lift, confidence, support) y por valores de variables, visualiza la dispersion
support vs confidence y revisa las reglas en una tabla paginada.

Ejecutar:
    .venv/bin/streamlit run artifacts/visor_app.py
"""

from math import ceil
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Resolucion de la raiz del repo: subir hasta encontrar el dir que contiene data/.
REPO = Path.cwd().resolve()
while not (REPO / 'data').is_dir() and REPO != REPO.parent:
    REPO = REPO.parent

CSV_REGLAS = REPO / 'data' / 'processed' / 'reglas_todas.csv'

NUMERICAS = [
    'age', 'gaming_hours', 'study_hours', 'sleep_hours', 'attendance',
    'social_activity', 'device_usage', 'reaction_time_ms', 'addiction_score',
    'grades',
]
CATEGORICAS = ['gender', 'gaming_genre', 'stress_level']
CUARTILES = ['Q1', 'Q2', 'Q3', 'Q4']

st.set_page_config(layout='wide', page_title='Visor de reglas de asociacion')
st.title('Visor de reglas de asociacion — Gaming vs Academic')


def _parsear_regla(antecedente, consecuente):
    """Convierte ambos lados de una regla en {variable: valor}."""
    valores = {}
    for lado in (antecedente, consecuente):
        for item in lado.split(' + '):
            var, val = item.split('=', 1)
            valores[var] = val
    return valores


@st.cache_data
def cargar_reglas():
    """Lee el CSV y agrega una columna ancha por variable (NaN si la regla no la usa)."""
    df = pd.read_csv(CSV_REGLAS)
    anchas = pd.DataFrame(
        [_parsear_regla(a, c) for a, c in zip(df['antecedente'], df['consecuente'])]
    )
    return pd.concat([df.reset_index(drop=True), anchas], axis=1)


df = cargar_reglas()
total = len(df)

# --- Sidebar: metricas (siempre aplicadas) ---
st.sidebar.header('Metricas')
lift_min, lift_max = float(df['lift'].min()), float(df['lift'].max())
sup_min, sup_max = float(df['support'].min()), float(df['support'].max())

rango_lift = st.sidebar.slider('lift', lift_min, lift_max, (lift_min, lift_max))
rango_conf = st.sidebar.slider('confidence', 0.0, 1.0, (0.0, 1.0))
rango_sup = st.sidebar.slider('support', sup_min, sup_max, (sup_min, sup_max))

# --- Sidebar: filtros opt-in por variable ---
st.sidebar.header('Variables (marcar para filtrar)')
filtros_num = {}
filtros_cat = {}

for var in NUMERICAS:
    if var in df.columns and st.sidebar.checkbox(var):
        filtros_num[var] = st.sidebar.select_slider(
            var, options=CUARTILES, value=('Q1', 'Q4')
        )

for var in CATEGORICAS:
    if var in df.columns and st.sidebar.checkbox(var):
        opciones = sorted(df[var].dropna().unique())
        filtros_cat[var] = st.sidebar.multiselect(var, options=opciones, default=opciones)

# --- Aplicacion de filtros ---
mask = (
    df['lift'].between(*rango_lift)
    & df['confidence'].between(*rango_conf)
    & df['support'].between(*rango_sup)
)

for var, (lo, hi) in filtros_num.items():
    permitidos = CUARTILES[CUARTILES.index(lo):CUARTILES.index(hi) + 1]
    mask &= df[var].isin(permitidos)  # NaN.isin(...) es False: solo reglas que usan la variable

for var, seleccion in filtros_cat.items():
    mask &= df[var].isin(seleccion)

filtrado = df[mask]

# --- Panel principal ---
st.caption(f'{len(filtrado)} de {total} reglas')

if filtrado.empty:
    st.info('Ninguna regla cumple los filtros seleccionados.')
else:
    fig = px.scatter(
        filtrado,
        x='support',
        y='confidence',
        color='lift',
        color_continuous_scale='Viridis',
        hover_data=['antecedente', 'consecuente'],
        height=500,
    )
    st.plotly_chart(fig, width='stretch')

    cols_tabla = ['antecedente', 'consecuente', 'support', 'confidence', 'lift']
    page_size = st.selectbox('Filas por pagina', [25, 50, 100, 200], index=1)
    n_pages = max(1, ceil(len(filtrado) / page_size))
    pagina = st.number_input('Pagina', 1, n_pages, 1)

    inicio = (pagina - 1) * page_size
    pagina_df = filtrado.iloc[inicio:inicio + page_size][cols_tabla].round(
        {'support': 3, 'confidence': 3, 'lift': 3}
    )
    st.dataframe(pagina_df, width='stretch', hide_index=True)
