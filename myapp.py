from pathlib import Path, PurePosixPath
from urllib.parse import quote

import dash
import dash_bootstrap_components as dbc
import flask


def create_app(app_name, app_filename, development_mode=False):
    #external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app_path = Path(app_filename)
    app_url = PurePosixPath('/', app_path.parent.name, app_path.name)
    server = flask.Flask(app_name)

    @server.after_request
    def add_header(response):
        if 'Cache-Control' not in response.headers:
            response.headers['Cache-Control'] = 'no-cache'
        return response

    server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets,
                    requests_pathname_prefix=quote(str(app_url) + "/"))
    app.title = "srMLGenes"
    return app
