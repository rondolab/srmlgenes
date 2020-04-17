import os, logging
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from functools import lru_cache

os.environ["WERKZEUG_DEBUG_PIN"] = "off"

likelihood_files = { ('kde_nearest', 'tennessen') : "ExAC_kde_inference_nearest.20200130.tsv",
                     ('kde_nearest', 'supertennessen') : "ExAC_63K_kde_nearest_nolscale_supertennessen_inference.tsv",
                     ('kde_nearest', 'subtennessen') : "kde_nearest_nolscale_exac_inference_subtennessen.tsv",
                     ('kde_3d', 'tennessen') : "ExAC_kde_inference_3d.20200124.tsv",
                     ('kde_3d', 'supertennessen') : "ExAC_63K_kde_3d_nolscale_supertennessen_inference.tsv",
                     ('kde_3d', 'subtennessen') : "kde_3d_nolscale_exac_inference_subtennessen.tsv",
                     ('prf', 'tennessen') : "ExAC_prf_inference.20191212.tsv",
                     ('prf', 'supertennessen') : "ExAC_63K_prf_supertennessen_inference.tsv",
                     ('prf', 'subtennessen') : "ExAC_63K_prf_subtennessen_inference.tsv" }

base_dir = "/sc/arion/projects/GENECAD/04_dominance"

genesets = {}
for name in "CGD_AR", "CGD_AD", "inbred_ALL", "haplo_Hurles_80", "haplo_Hurles_low20":
    filename = os.path.join(base_dir, 'slim/mock_genome', name + '.tsv')
    geneset = set()
    with open(filename) as list_file:
        for gene in list_file:
            if gene.strip() != "gene":
                geneset.add(gene.strip())
    genesets[name] = geneset


@lru_cache(maxsize=None)
def load_unfiltered_df(likelihood, demography):
    filename = os.path.join(base_dir, "genedose", likelihood_files[likelihood, demography])
    return pd.read_table(filename)


def filter_df(df, func, genelist, min_L, max_L):
    selector = df.func == func
    selector &= df.U.between(10**(min_L-8), 10**(max_L-8))
    if genelist != "all":
        selector &= df.gene.isin(genesets[genelist])
    return df.loc[selector]


def format_heatmap_empirical(filtered_df):
    crosstab = pd.crosstab(filtered_df["h_grid17_ml"],
                           filtered_df["s_grid17_ml"],
                           dropna=False, margins=True)
    counts_grid = []
    for h in 0.0, 0.1, 0.3, 0.5:
        if h == 0.5:
            counts_row = [crosstab.loc["All", 0.0]]
        else:
            counts_row = [None]
        for s in 1e-4, 1e-3, 1e-2, 1e-1:
            try:
                counts_row.append(crosstab.loc[h, s])
            except KeyError:
                counts_row.append(0)
        counts_grid.append(counts_row)
    return counts_grid


@lru_cache(maxsize=None)
def load_exac_data(likelihood, demography, func, genelist, min_L, max_L):
    unfiltered_df = load_unfiltered_df(likelihood, demography)
    filtered_df = filter_df(unfiltered_df, func, genelist, min_L, max_L)
    return format_heatmap_empirical(filtered_df)


def heatmap_figure(heatmap_data):
    heatmap_data = np.array(heatmap_data, dtype=float)
    total_genes = np.nansum(heatmap_data)
    percent = np.round(heatmap_data / total_genes * 100, 1)
    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=['Neutral', '-10⁻⁴', '-10⁻³', '-10⁻²', '-10⁻¹'],
                        y=["0.0", "0.1", "0.3", "0.5"],
                        customdata=percent,
                        hoverongaps=False,
                        hovertemplate=f"h: %{{y}}<br />s: %{{x}}<br />genes: %{{z}}/{total_genes:0.0f} (%{{customdata}}%)<extra></extra>"),
                    layout=go.Layout(width=800, height=600,
                xaxis_type='category', yaxis_type='category'))
    return fig


def make_heatmap(likelihood, demography, func, genelist, min_L, max_L):
    data = load_exac_data(likelihood, demography, func, genelist, min_L, max_L)
    return heatmap_figure(data)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                requests_pathname_prefix="/~jordad05/heatmaps/exac.wsgi/")

app.layout = html.Div(children=[
                dcc.Graph(id='heatmap', style={'height': '600px'}),
                dcc.Dropdown(id="geneset-dropdown",
                             options=[{'label': 'All genes', 'value': 'all'},
                                      {'label': 'HI > 80%', 'value': 'haplo_Hurles_80'},
                                      {'label': 'CGD AD', 'value': 'CGD_AD'},
                                      {'label': 'Inbred', 'value': 'inbred_ALL'},
                                      {'label': 'HI < 20%', 'value': 'haplo_Hurles_low20'},
                                      {'label': 'CGD AR', 'value': 'CGD_AR'}],
                             value='all', style={'width': '7em', 'display': 'inline-block'}),
                dcc.Dropdown(id="likelihood-dropdown",
                             options=[{'label': 'PRF', 'value': 'prf'},
                                      {'label': 'KDE (nearest)', 'value': 'kde_nearest'},
                                      {'label': 'KDE (3D)', 'value': 'kde'}],
                             value='prf', style={'width': '9em', 'display': 'inline-block'}),
                dcc.Dropdown(id="demography-dropdown",
                             options=[{'label': 'Tennessen reference', 'value': 'tennessen'},
                                      {'label': 'Super-Tennessen reference', 'value': 'supertennessen'}],
                             value='tennessen', style={'width': '15em', 'display': 'inline-block'}),
                dcc.Dropdown(id="func-dropdown",
                             options=[{'label': "LOF + PolyPhen probably", 'value': 'LOF_probably'},
                                      {'label': "synonymous", 'value': 'synon'}],
                             value="LOF_probably", style={'width': '15em', 'display': 'inline-block'}),
                html.Label("L"), dcc.RangeSlider(id="L-slider", min=0, max=6, step=0.1,
                    marks={0: '10⁰', 1: '10¹', 2: '10²', 3 : '10³', 4: '10⁴', 5: '10⁵', 6: '10⁶'},
                                            value=[2,5],
                                            tooltip={'always_visible' : False})
                ], style={'width': '800px'})
application = app.server


@app.callback(Output('heatmap', 'figure'), 
              [Input('likelihood-dropdown', 'value'),
               Input('demography-dropdown', 'value'),
               Input('func-dropdown', 'value'),
               Input('geneset-dropdown', 'value'),
               Input('L-slider', 'value')])
def update_heatmap(likelihood, demography, func, geneset, Ls):
    return make_heatmap(likelihood, demography, func, geneset, Ls[0], Ls[1])


if __name__ == "__main__":
    app.run_server(debug=True)
