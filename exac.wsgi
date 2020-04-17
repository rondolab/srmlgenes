import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from heatmaps_data import load_exac_data, heatmap_figure

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
