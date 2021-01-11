import os
import sys

import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

sys.path.append(os.path.dirname(__file__))
from heatmaps_common import create_app, gene_select_controls, make_heatmap_single_sim, \
    make_heatmap_geneset_sim

app = create_app(__name__, __file__)

app.layout = html.Div(children=[
                dcc.Graph(id='heatmap', style={'height': '600px'}),
                html.Label("h"), dcc.Slider(id="h-slider", min=0, max=3, 
                                            marks={0: '0.0', 1: '0.1', 2: '0.3', 3: '0.5'},
                                            disabled=True, value=3),
                html.Label("s"), dcc.Slider(id="s-slider", min=0, max=4, 
                    marks={0: 'Neutral', 1: '-10⁻⁴', 2: '-10⁻³', 3: '-10⁻²', 4: '-10⁻¹'},
                                            value=0),
                dcc.RadioItems(id="L-select-mode",
                    options=[{'label': "Empirical length profile", 'value': 'empirical'},
                             {'label': "Single length", 'value': 'single'}], value='single'),
                html.Div(id="L-select-single", children=[html.Label("L"), dcc.Slider(id="L-slider-single", min=2, max=5, step=0.1,
                    marks={2: '10²', 3 : '10³', 4: '10⁴', 5: '10⁵', 6: '10⁶'},
                                            value=3,
                                            tooltip={'always_visible' : False})]),
                html.Div(id="L-select-empirical", children=gene_select_controls())
                ], style={'width': '800px'})
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


@app.callback(Output('heatmap', 'figure'), 
              [Input('h-slider', 'value'),
               Input('s-slider', 'value'),
               Input('func-dropdown', 'value'),
               Input('geneset-dropdown', 'value'),
               Input('L-slider', 'value'),
               Input('L-slider-single', 'value'),
               Input('L-select-mode', 'value')])
def update_heatmap(h_idx, s_idx, func, geneset, L_boundaries, single_L, L_mode):
    if L_mode == "single":
        return make_heatmap_single_sim("prf", "supertennessen", "supertennessen", s_labels[s_idx], h_labels[h_idx], single_L)
    elif L_mode == "empirical":
        L_boundaries = np.clip(L_boundaries, 2.0, 5.0)
        return make_heatmap_geneset_sim("prf", "supertennessen", "supertennessen", s_labels[s_idx], h_labels[h_idx],
                                        func, geneset, L_boundaries[0], L_boundaries[1])
    else:
        raise ValueError(f"Unknown L selection mode {L_mode}")


if __name__ == "__main__":
    app.config.update(requests_pathname_prefix="/")
    app.run_server(debug=True)
