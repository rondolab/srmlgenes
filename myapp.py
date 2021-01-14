from pathlib import Path, PurePosixPath
from urllib.parse import quote

import dash
import flask


def create_app(app_name, app_filename, development_mode=False):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app_path = Path(app_filename)
    app_url = PurePosixPath('/', app_path.parent.name, app_path.name)
    server = flask.Flask(app_name)
    server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets,
                    requests_pathname_prefix=quote(str(app_url) + "/"))
    if development_mode:
        app.enable_dev_tools(debug=True)
    return app
