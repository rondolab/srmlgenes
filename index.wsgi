import os
import sys

import dash_core_components as dcc

sys.path.append(os.path.dirname(__file__))
from heatmaps_common import create_app, SimsTab, ExacTab

app = create_app(__name__, __file__)

sim_tab = SimsTab()
exac_tab = ExacTab()

app.layout = dcc.Tabs(id="tabs", value="sims", children=[
                dcc.Tab(label="Simulated Genes", value='sims',
                        children=sim_tab.render_layout()),
                dcc.Tab(label="ExAC Genes", value='exac',
                        children=exac_tab.render_layout())])

sim_tab.register_callbacks(app)
exac_tab.register_callbacks(app)

application = app.server


if __name__ == "__main__":
    app.config.update(requests_pathname_prefix="/")
    app.run_server(debug=True)
