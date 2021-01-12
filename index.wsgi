import os
import sys

import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

sys.path.append(os.path.dirname(__file__))
from heatmaps_common import create_app, gene_select_controls, make_heatmap_single_sim, \
    make_heatmap_geneset_sim, make_heatmap_empirical

app = create_app(__name__, __file__)

app.layout = html.Div([
        dcc.Tabs(id="tabs", value="sims", children=[
        dcc.Tab(label="Simulated Genes", value='sims', children=[html.Div(children=[
                    html.Div(children=[
                        html.Label("h"),
                        dcc.Slider(id="h-slider", min=0, max=3,
                                    marks={0: '0.0', 1: '0.1', 2: '0.3', 3: '0.5'},
                                    disabled=True, value=3),
                    html.Br(),
                    html.Label("s"),
                        dcc.Slider(id="s-slider", min=0, max=4,
                                    marks={0: 'Neutral', 1: '-10⁻⁴', 2: '-10⁻³', 3: '-10⁻²', 4: '-10⁻¹'},
                                    value=0),
                    html.Br(),
                        dcc.RadioItems(id="L-select-mode",
                            options=[{'label': "Empirical length profile", 'value': 'empirical'},
                                     {'label': "Single length", 'value': 'single'}], value='single'),
                    html.Div(id="L-select-single", children=[
                        html.Label("L"),
                        dcc.Slider(id="L-slider-single", min=2, max=5, step=0.1,
                                  marks={2: '10²', 3 : '10³', 4: '10⁴', 5: '10⁵'},
                                    value=3,
                                    tooltip={'always_visible' : False})]),
                    html.Div(id="L-select-empirical", children=gene_select_controls("-sim"))
                ], style={'width': '30%',
                          'margin-left': '5%',
                          'margin-right': '5%',
                          'margin-top': '10%',
                          'display': 'inline-block'}),
                    html.Div(children=[
                        dcc.Graph(id='heatmap-sim')
                    ], style={'width': '60%',
                              'display': 'inline-block',
                              'float': 'right'})],
                style={ 'width': '800px' })]),
        dcc.Tab(label="ExAC Genes", value='exac', children=[html.Div(children=[
                html.Div([html.Label("Color Scheme"),
                dcc.RadioItems(id="color-buttons",
                               options=[{'label': 'Histogram', 'value': 'histogram'},
                                        {'label': 'Enrichment (log odds ratio)', 'value': 'odds_ratio'},
                                        {'label': 'Enrichment (p-value)', 'value': 'p_value'}],
                               value='histogram')] +
                gene_select_controls("-exac"),
                style={'width': '30%',
                          'margin-left': '5%',
                          'margin-right': '5%',
                          'margin-top': '10%',
                          'display': 'inline-block'}),
                html.Div([dcc.Graph(id='heatmap-exac', style={'height': '600px'})],
                         style={'width': '60%',
                              'display': 'inline-block',
                              'float': 'right'})],
                 style={'width': '800px'})])])])
application = app.server


@app.callback([Output('L-select-single', 'hidden'), Output('L-select-empirical', 'hidden')],
              [Input('L-select-mode', 'value')])
def switch_L_selection_visibility(mode):
    if mode == "empirical":
        return True, False
    elif mode == "single":
        return False, True
    else:
        raise ValueError(f"Unrecognized L selection mode {mode}")


@app.callback(Output('h-slider', 'disabled'),
              [Input('s-slider', 'value')])
def adjust_h_slider(s_slider_value):
    return s_slider_value == 0

h_labels = ["0.0", "0.1", "0.3", "0.5"]
s_labels = ["NEUTRAL", "-4.0", "-3.0", "-2.0", "-1.0"]


@app.callback(Output('heatmap-sim', 'figure'),
              [Input('h-slider', 'value'),
               Input('s-slider', 'value'),
               Input('func-dropdown-sim', 'value'),
               Input('geneset-dropdown-sim', 'value'),
               Input('L-slider-sim', 'value'),
               Input('L-slider-single', 'value'),
               Input('L-select-mode', 'value')])
def update_heatmap_sim(h_idx, s_idx, func, geneset, L_boundaries, single_L, L_mode):
    if s_idx == 0:
        h_idx = 3
    if L_mode == "single":
        return make_heatmap_single_sim("prf", "supertennessen", "supertennessen", s_labels[s_idx], h_labels[h_idx], single_L)
    elif L_mode == "empirical":
        L_boundaries = np.clip(L_boundaries, 2.0, 5.0)
        return make_heatmap_geneset_sim("prf", "supertennessen", "supertennessen", s_labels[s_idx], h_labels[h_idx],
                                        func, geneset, L_boundaries[0], L_boundaries[1])
    else:
        raise ValueError(f"Unknown L selection mode {L_mode}")


@app.callback(Output('heatmap-exac', 'figure'),
              [Input('func-dropdown-exac', 'value'),
               Input('geneset-dropdown-exac', 'value'),
               Input('L-slider-exac', 'value'),
               Input('color-buttons', 'value')])
def update_heatmap_empirical(func, geneset, Ls, z_variable):
    return make_heatmap_empirical("prf", "supertennessen", func, geneset, Ls[0], Ls[1], z_variable)


if __name__ == "__main__":
    app.config.update(requests_pathname_prefix="/")
    app.run_server(debug=True)
