import os
import sys

import tables
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

sys.path.append(os.path.dirname(__file__))
from heatmaps_common import create_app, gene_select_controls, make_heatmap_empirical

app = create_app(__name__, __file__)

app.layout = html.Div(children=[
                dcc.Graph(id='heatmap', style={'height': '600px'}),
                html.Label("Color Scheme"),
                dcc.RadioItems(id="color-buttons",
                               options=[{'label': 'Histogram', 'value': 'histogram'},
                                        {'label': 'Enrichment (log odds ratio)', 'value': 'odds_ratio'},
                                        {'label': 'Enrichment (p-value)', 'value': 'p_value'}],
                               value='histogram')] +
                gene_select_controls(), style={'width': '800px'})
application = app.server


@app.callback(Output('heatmap', 'figure'), 
              [Input('likelihood-dropdown', 'value'),
               Input('demography-dropdown', 'value'),
               Input('func-dropdown', 'value'),
               Input('geneset-dropdown', 'value'),
               Input('L-slider', 'value'),
               Input('color-buttons', 'value')])
def update_heatmap(likelihood, demography, func, geneset, Ls, z_variable):
    return make_heatmap_empirical(likelihood, demography, func, geneset, Ls[0], Ls[1], z_variable)


if __name__ == "__main__":
    app.config.update(requests_pathname_prefix="/")
    app.run_server(debug=True)
