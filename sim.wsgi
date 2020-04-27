import sys, os
sys.path.append(os.path.dirname(__file__))

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from heatmaps_common import heatmap_figure, load_sim_data, create_app, gene_select_controls


def make_heatmap(likelihood, ref, sim, s, h, L):
    data = load_sim_data(likelihood, ref, sim, s, h, L)
    return heatmap_figure(data)


app = create_app(__name__, __file__)

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
                dcc.RadioItems(id="L-select-mode",
                    options=[{'label': "Empirical gene set", 'value': 'empirical'},
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
              [Input('likelihood-dropdown', 'value'),
               Input('ref-dropdown', 'value'),
               Input('sim-dropdown', 'value'),
               Input('h-slider', 'value'),
               Input('s-slider', 'value'),
               Input('L-slider-single', 'value')])
def update_heatmap(likelihood, ref, sim, h_idx, s_idx, L):
    return make_heatmap(likelihood, ref, sim,
                        s_labels[s_idx],
                        h_labels[h_idx],
                        f'{L:0.1f}')
if __name__ == "__main__":
    app.run_server(debug=True)
