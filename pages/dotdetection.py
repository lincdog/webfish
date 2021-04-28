import os
import json
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, config, s3_client
import cloud

data_manager = cloud.DataManager(config=config, s3_client=s3_client, pagename='dotdetection')
data_manager.find_datafiles()


def get_tree():
    fname = data_manager.request(None, fields='exp_tree', force_download=True)

    return json.load(open(fname['exp_tree'], 'r'))


exp_tree = get_tree()


@app.callback(
    Output('dataset-wrapper', 'children'),
    Input('user-select', 'value')
)
def select_user(user):
    datasets = list(exp_tree.get(user, {}).keys())

    return dcc.Dropdown(id='dataset-select',
                        value=None,
                        options=[{'label': d, 'value': d} for d in sorted(datasets)])


@app.callback(
    Output('analysis-wrapper', 'children'),
    Input('dataset-select', 'value'),
    State('user-select', 'value')
)
def select_dataset(dataset, user):
    try:
        analyses = ['New'] + exp_tree[user][dataset]
    except KeyError:
        raise PreventUpdate

    return dcc.Dropdown(id='analysis-select', value='New',
                        options=[{'label': i, 'value': i} for i in sorted(analyses)])


layout = html.Div([
    dbc.Alert('In this tab you can preview the results of dot detection'
                  ' using various parameter settings. Data is synced from the'
                  ' HPC analyses folder every 2 minutes.', color='info'),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='user-select', options=[{'label': i, 'value': i}
                                                    for i in sorted(exp_tree.keys())]),
            html.Div(dcc.Dropdown(id='dataset-select'), id='dataset-wrapper'),
            html.Div(dcc.Dropdown(id='analysis-select'), id='analysis-wrapper'),
            dcc.Slider(id='strictness-slider')
        ], width=4)
    ])
])
