import os
import sys

import tables
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

sys.path.append(os.path.dirname(__file__))
from heatmaps_common import heatmap_figure, create_app, gene_select_controls

tables_file = tables.open_file(os.path.join(os.path.dirname(__file__), "heatmaps.hdf5"))

def make_heatmap(likelihood, demography, func, genelist, min_L, max_L):
    data_group = tables_file.get_node(f"/exac/{likelihood}/{demography}/{func}/{genelist}/{min_L}/{max_L}")
    return heatmap_figure(data_group)


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
