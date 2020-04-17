import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from functools import lru_cache

sim_data_template = "~/genecad/04_dominance/genedose/simulation_inference_{likelihood}/{likelihood}_ref_{ref}_sims_{sim}_S_{s}_h_{h}_L_{L}.tsv"
exac_sumstats = pd.read_table("/hpc/users/jordad05/genecad/04_dominance/genedose/ExAC_63K_symbol_plus_ensembl_func_summary_stats.tsv")
func_length_tables = {}
for func in 'LOF_probably', 'synon':
    func_length_tables[func] = exac_sumstats.loc[exac_sumstats.func == func, "L"]\
                                            .transform('log10')\
                                            .round(1)\
                                            .clip(2.0, 5.0)\
                                            .value_counts()

def format_heatmap_sims(df):
    ml_bin_names = df.transpose().drop("L").idxmax()
    split_names = ml_bin_names.str.split("_")
    ml_s = split_names.str.get(0)
    ml_h = split_names.str.get(1)
    crosstab = pd.crosstab(ml_h, ml_s)
    counts_grid = crosstab.reindex(["0.0", "0.1", "0.3", "0.5"], fill_value=0, axis=0)\
                          .reindex(["NEUTRAL", "-4.0", "-3.0", "-2.0", "-1.0"], fill_value=0, axis=1)\
                          .values.tolist()
    for h_index in 0, 1, 2:
        counts_grid[h_index][0] = None
    return counts_grid


@lru_cache(maxsize=None)
def load_sim_data(likelihood, ref, sim, s, h, L):
    if L in func_length_tables:
        sims_to_concat = []
        for l, count in func_length_tables[L].iteritems():
            sims_to_concat.append(pd.read_table(sim_data_template.format(likelihood=likelihood,
                                                                         ref=ref,
                                                                         sim=sim,
                                                                         s=s,
                                                                         h=h,
                                                                         L=l),
                                                nrows=count))
        df = pd.concat(sims_to_concat, ignore_index=True)
    else:
        df = pd.read_table(sim_data_template.format(likelihood=likelihood,
                                                    ref=ref,
                                                    sim=sim,
                                                    s=s, h=h, L=L))
    return format_heatmap_sims(df)


def heatmap_figure(heatmap_data):
    total_genes = np.nansum(np.array(heatmap_data, dtype=float))
    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=['Neutral', '-10⁻⁴', '-10⁻³', '-10⁻²', '-10⁻¹'],
                        y=["0.0", "0.1", "0.3", "0.5"],
                        hoverongaps=False,
                        hovertemplate="h: %%{y}<br />s: %%{x}<br />genes: %%{z}/%d<extra></extra>" % total_genes),
                    layout=go.Layout(width=800, height=600,
                xaxis_type='category', yaxis_type='category'))
    return fig

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

def make_heatmap(likelihood, ref, sim, s, h, L):
    data = load_sim_data(likelihood, ref, sim, s, h, L)
    return heatmap_figure(data)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                requests_pathname_prefix="/~jordad05/heatmaps/sim.wsgi/")

app.layout = html.Div(children=[
                dcc.Graph(id='heatmap', style={'height': '600px'}),
                dcc.Dropdown(id="likelihood-dropdown",
                             options=[{'label': 'PRF', 'value': 'prf'},
                                      {'label': 'KDE (nearest)', 'value': 'kde_nearest'},
                                      {'label': 'KDE (3D)', 'value': 'kde'}],
                             value='prf', style={'width': '12em', 'display': 'inline-block'}),
                dcc.Dropdown(id="ref-dropdown",
                             options=[{'label': 'Tennessen reference', 'value': 'tennessen'},
                                      {'label': 'Super-Tennessen reference', 'value': 'supertennessen'}],
                             value='tennessen', style={'width': '20em', 'display': 'inline-block'}),
                dcc.Dropdown(id="sim-dropdown",
                             options=[{'label': 'Tennessen simulations', 'value': 'tennessen'},
                                      {'label': 'Super-Tennessen simulations', 'value': 'supertennessen'}],
                             value='tennessen', style={'width' : '20em', 'display': 'inline-block'}),
                html.Label("h"), dcc.Slider(id="h-slider", min=0, max=3, 
                                            marks={0: '0.0', 1: '0.1', 2: '0.3', 3: '0.5'},
                                            disabled=True, value=3),
                html.Label("s"), dcc.Slider(id="s-slider", min=0, max=4, 
                    marks={0: 'Neutral', 1: '-10⁻⁴', 2: '-10⁻³', 3: '-10⁻²', 4: '-10⁻¹'},
                                            value=0),
                #html.Label("L"), dcc.Slider(id="L-slider", min=0, max=6,
                #    marks={0: 'synon', 1: 'LOF_probably', 
                #           2: '100', 3: '300', 4: '1000', 5: '3000', 6: '10000' },
                #                            value=1)
                #html.Label("L"), html.RadioItems(id="L-select-style",
                #    options=[{'label': "Empirical gene set", 'value': 'empirical'},
                #             {'label': "Single length", 'value': 'single'}, value='single'] 
                # html.Div(id="L-select")
                dcc.Slider(id="L-slider", min=2, max=5, step=0.1,
                    marks={2: '10²', 3 : '10³', 4: '10⁴', 5: '10⁵', 6: '10⁶'},
                                            value=3,
                                            tooltip={'always_visible' : False})
                ], style={'width': '800px'})
application = app.server

#@app.callback(Output('L-select', 'children'),
#              [Input('L-select-style', 'value')])
#def choose_L_selector(

@app.callback([Output('h-slider', 'value'), Output('h-slider', 'disabled')],
              [Input('s-slider', 'value')],
              [State('h-slider', 'value')])
def adjust_h_slider(s_slider_value, h_slider_value):
    if s_slider_value == 0:
        return 3, True
    else:
        return h_slider_value, False

h_labels = ["0.0", "0.1", "0.3", "0.5"]
s_labels = ["NEUTRAL", "-4.0", "-3.0", "-2.0", "-1.0"]
#L_labels = ["LOF_probably", "synon", "2.0", "2.5", "3.0", "3.5", "4.0"]


@app.callback(Output('heatmap', 'figure'), 
              [Input('likelihood-dropdown', 'value'),
               Input('ref-dropdown', 'value'),
               Input('sim-dropdown', 'value'),
               Input('h-slider', 'value'),
               Input('s-slider', 'value'),
               Input('L-slider', 'value')])
def update_heatmap(likelihood, ref, sim, h_idx, s_idx, L):
    return make_heatmap(likelihood, ref, sim,
                        s_labels[s_idx],
                        h_labels[h_idx],
                        f'{L:0.1f}')


if __name__ == "__main__":
    app.run_server(debug=True)
