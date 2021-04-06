import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as db
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, config, s3_client
from pages import datavis, dotdetection

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.H1('webfish app'),
    dcc.Tabs(id='main-tabs', value='datavis', children=[
        dcc.Tab(label='Data Visualization', value='datavis'),
        dcc.Tab(label='Dot detection preview', value='dotdetection')
    ], style={'width': '500px'}),
    html.Div(id='content-main')
], style={'margin': 'auto',})

@app.callback(
    Output('content-main', 'children'),
    Input('main-tabs', 'value')
)
def tab_handler(tabval):
    if tabval == 'datavis':
        return datavis.layout
    elif tabval == 'dotdetection':
        return dotdetection.layout
    else:
        return html.H1('404!!!!')


if __name__ == '__main__':
    hostip = os.environ.get('WEBFISH_HOST', '127.0.0.1')
    hostport = os.environ.get('WEBFISH_PORT', '8050')
    app.run_server(debug=True, host=hostip, port=hostport)
