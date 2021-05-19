import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, config, s3_client
from pages import datavis, dotdetection

valid_pagenames = tuple(config['pages'].keys())

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='wf-store', storage_type='session', data={}),
    html.H1('webfish app'),
    dcc.Tabs(id='main-tabs', children=[
        dcc.Tab(
            label='Data Visualization',
            value='datavis',
            #children=datavis.layout
        ),
        dcc.Tab(
            label='Dot detection preview',
            value='dotdetection',
            #children=dotdetection.layout
        )
    ], style={'width': '500px'}),
    html.Div(id='content-main', style={'width': '100%', 'height': '100%'})
], style={'margin': 'auto'})


@app.callback(
    Output('content-main', 'children'),
    Input('main-tabs', 'value'),
)
def tab_handler(tabval):
    if tabval == 'datavis':
        return datavis.layout
    elif tabval == 'dotdetection':
        return dotdetection.layout
    else:
        return html.H1('404!!!!')


@app.callback(
    Output('url', 'pathname'),
    Output('main-tabs', 'value'),
    Input('url', 'pathname'),
    Input('main-tabs', 'value')
)
def sync_tab_url(pathname, tabval):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    pathname = pathname.strip('/')

    if trigger_id == 'url' or tabval not in valid_pagenames:
        newval = pathname
    else:
        newval = tabval

    if newval == '':
        newval = 'datavis'  # default page

    if newval not in valid_pagenames:
        raise PreventUpdate

    return newval, newval


if __name__ == '__main__':
    hostip = os.environ.get('WEBFISH_HOST', '127.0.0.1')
    hostport = os.environ.get('WEBFISH_PORT', '8050')
    app.run_server(
        debug=True,
        host=hostip,
        port=hostport,
        #dev_tools_props_check=False
    )
