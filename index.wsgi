import os
import sys

import dash_core_components as dcc

sys.path.append(os.path.dirname(__file__))
from myapp import create_app
from layouts import CacheBuster, TwoTabLayout

app = create_app(__name__, __file__, development_mode=("dev" in __file__))

layout_obj = CacheBuster(TwoTabLayout())
layout_obj.attach_to_app(app)

application = app.server


if __name__ == "__main__":
    app.config.update(requests_pathname_prefix="/")
    app.run_server(debug=True)
