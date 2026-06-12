"""Genera un visor HTML interactivo de las reglas de asociación.

Lee `data/processed/reglas_todas.csv` (salida del notebook 03) y produce
`artifacts/visor_reglas.html`: un scatter Plotly soporte × confianza, color = lift,
con hover que muestra la regla y un desplegable para filtrar por variable.

Uso:
    .venv/bin/python artifacts/build_visor.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# raíz del repo: subir hasta el directorio que contiene data/ (funciona desde la raíz o desde artifacts/)
REPO = Path.cwd().resolve()
while not (REPO / 'data').is_dir() and REPO != REPO.parent:
    REPO = REPO.parent
ENTRADA = REPO / 'data' / 'processed' / 'reglas_todas.csv'
SALIDA = REPO / 'artifacts' / 'visor_reglas.html'

reglas = pd.read_csv(ENTRADA)


def variables_de(regla):
    """Conjunto de variables (sin el =valor) que toca una regla, ambos lados."""
    items = f"{regla['antecedente']} + {regla['consecuente']}".split(' + ')
    return {it.split('=')[0] for it in items}


reglas['variables'] = reglas.apply(variables_de, axis=1)
todas_vars = sorted({v for s in reglas['variables'] for v in s})

hover = (
    '<b>' + reglas['antecedente'] + '</b><br>→ ' + reglas['consecuente']
    + '<br>soporte=' + reglas['support'].round(3).astype(str)
    + ' · confianza=' + reglas['confidence'].round(3).astype(str)
    + ' · lift=' + reglas['lift'].round(3).astype(str)
    + '<extra></extra>'
)
reglas['hover'] = hover

lmin, lmax = reglas['lift'].min(), reglas['lift'].max()


def traza(sub, nombre, visible):
    return go.Scattergl(
        x=sub['support'], y=sub['confidence'],
        mode='markers', name=nombre, visible=visible,
        marker=dict(
            size=6, opacity=0.55, color=sub['lift'], colorscale='Viridis',
            cmin=lmin, cmax=lmax, showscale=True,
            colorbar=dict(title='lift'),
            line=dict(width=0),
        ),
        hovertemplate=sub['hover'],
    )


# traza 0 = todas; luego una por variable (oculta)
trazas = [traza(reglas, f'Todas ({len(reglas)})', True)]
subsets = [reglas]
for v in todas_vars:
    sub = reglas[reglas['variables'].apply(lambda s: v in s)]
    trazas.append(traza(sub, f'{v} ({len(sub)})', False))
    subsets.append(sub)

n = len(trazas)
botones = []
for i, (nombre, sub) in enumerate(zip(
        ['Todas las reglas'] + todas_vars, subsets)):
    vis = [j == i for j in range(n)]
    botones.append(dict(
        label=f'{nombre} ({len(sub)})', method='update',
        args=[{'visible': vis},
              {'title': f'Reglas de asociación — {nombre} ({len(sub)} reglas)'}],
    ))

fig = go.Figure(data=trazas)
fig.update_layout(
    title=f'Reglas de asociación — Todas las reglas ({len(reglas)} reglas)',
    xaxis_title='soporte', yaxis_title='confianza',
    template='plotly_white', height=720,
    updatemenus=[dict(
        buttons=botones, direction='down', showactive=True,
        x=1.02, xanchor='left', y=1, yanchor='top',
    )],
    annotations=[dict(
        text='Filtrar por variable →', x=1.02, xref='paper', xanchor='left',
        y=1.06, yref='paper', showarrow=False, font=dict(size=12),
    )],
)

SALIDA.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(SALIDA, include_plotlyjs=True, full_html=True)
print('Guardado:', SALIDA)
print('Reglas:', len(reglas), '| variables:', len(todas_vars))
