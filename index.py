import logging
logging.basicConfig(level="DEBUG")

logging.debug("This is a debug message from the web app")

import os
import sys

#app_dir = os.path.dirname(__file__)
#virtualenv_activate_script = os.path.join(app_dir, "venv", "bin", "activate_this.py")
#exec(open(virtualenv_activate_script).read(), {'__file__': virtualenv_activate_script})
#sys.path.append(app_dir)

import dash_core_components as dcc

from myapp import create_app
from layouts import TwoTabLayout
logging.debug("Succesffully loaded layouts")

app = create_app(__name__, __file__, development_mode=("dev" in __file__))
logging.debug("Succesffully created app")

layout_obj = TwoTabLayout()
layout_obj.attach_to_app(app)
logging.debug("Succesffully attached layout to app")

application = app.server

#logging.debug("Running server...")
#app.run_server()
