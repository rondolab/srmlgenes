import sys, os
sys.path.append(os.path.dirname(__file__))

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from heatmaps_common import load_exac_data, heatmap_figure, create_app, gene_select_controls


def make_heatmap(likelihood, demography, func, genelist, min_L, max_L):
    histogram, odds_ratios, pvalues = load_exac_data(likelihood, demography, func, genelist, min_L, max_L)
    if odds_ratios is None:
        return heatmap_figure(histogram)
    else:
        return heatmap_figure(histogram, odds_ratios, pvalues)


app = create_app(__name__, __file__)

app.layout = html.Div(children=[
                dcc.Graph(id='heatmap', style={'height': '600px'}),
                dcc.Dropdown(id="likelihood-dropdown",
                             options=[{'label': 'PRF', 'value': 'prf'},
                                      {'label': 'KDE (nearest)', 'value': 'kde_nearest'},
                                      {'label': 'KDE (3D)', 'value': 'kde'}],
                             value='prf', style={'width': '9em', 'display': 'inline-block'}),
                dcc.Dropdown(id="demography-dropdown",
                             options=[{'label': 'Tennessen reference', 'value': 'tennessen'},
                                      {'label': 'Super-Tennessen reference', 'value': 'supertennessen'}],
                             value='tennessen', style={'width': '15em', 'display': 'inline-block'})] +\
                gene_select_controls(), style={'width': '800px'})
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
