import os
import sys

app_dir = os.path.dirname(__file__)
virtualenv_activate_script = os.path.join(app_dir, "venv", "bin", "activate_this.py")
exec(open(virtualenv_activate_script).read(), {'__file__': virtualenv_activate_script})
sys.path.append(app_dir)

import dash_core_components as dcc

from myapp import create_app
from layouts import TwoTabLayout

app = create_app(__name__, __file__, development_mode=("dev" in __file__))

layout_obj = TwoTabLayout()
layout_obj.attach_to_app(app)

application = app.server


if __name__ == "__main__":
#    app.config.update(requests_pathname_prefix="/")
    app.run_server()
