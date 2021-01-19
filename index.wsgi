import os
import sys

import dash_core_components as dcc

app_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(app_dir, "venv", "lib", "python3.8", "site-"))
sys.path.append(app_dir)
from myapp import create_app
from layouts import TwoTabLayout

app = create_app(__name__, __file__, development_mode=("dev" in __file__))

layout_obj = TwoTabLayout()
layout_obj.attach_to_app(app)

application = app.server


if __name__ == "__main__":
    app.config.update(requests_pathname_prefix="/")
    app.run_server(debug=True)
