import numpy as np
import pandas as pd
import io
import json
import re

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import no_update

from app import app
from .common import ComponentManager, data_clients

data_client = data_clients['__all__']


# TODO: move this to DataClient
def put_analysis_request(
    user,
    dataset,
    analysis_name,
    dot_detection='biggest jump 3d'
):
    analysis_dict = {
        'personal': user,
        'experiment_name': dataset,
        'dot detection': dot_detection,
        'dot detection test': 'true',
        'visualize dot detection': 'true',
        'strictness': 'multiple',
        'clusters': {
            'ntasks': '1',
            'mem-per-cpu': '10G',
            'email': 'nrezaee@caltech.edu'
        }
    }

    dict_bytes = io.BytesIO(json.dumps(analysis_dict).encode())
    # These are "characters to avoid" in keys according to AWS docs
    # \, {, }, ^, [, ], %, `, <, >, ~, #, |
    # TODO: Sanitize filenames on server side before uploading too; mostly
    #   potential problem for user-defined dataset/analysis names
    analysis_sanitized = re.sub('[\\\\{^}%` \\[\\]>~<#|]', '', analysis_name)
    keyname = f'json_analyses/{analysis_sanitized}.json'

    try:
        data_client.client.client.upload_fileobj(
            dict_bytes,
            Bucket=data_client.bucket_name,
            Key=keyname
        )
    except Exception as e:
        return str(e)

    print(f'analysis_dict: {json.dumps(analysis_dict, indent=2)}')

    return analysis_sanitized


clear_components = {
    'sb-analysis-name':
        dbc.Input(
            id='sb-analysis-name',
            placeholder='Type a name for the analysis'
        ),
    'sb-alignment-select':
        dbc.Select(
            id='sb-alignment-select',
            value='mean squares 2d',
            options=[{'label': 'Mean Squares 2D', 'value': 'mean squares 2d'}],
            disabled=True
        ),
    'sb-dot-detection-select':
        dbc.Select(
            id='sb-dot-detection-select',
            value='biggest jump 3d',
            options=[{'label': 'Biggest Jump 3D', 'value':'biggest jump 3d'}],
            disabled=True
        ),
    'sb-strictness-select':
        dbc.Input(
            id='sb-strictness-select',
            type='number',
            min=-10,
            max=15,
            step=1,
            value=2
        ),
    'sb-channel-select':
        dbc.Checklist(
            options=[
                {'label': '1', 'value': '1'},
                {'label': '2', 'value': '2'},
                {'label': '3', 'value': '3'}
            ],
            id='sb-channel-select',
            inline=True
        ),
    'sb-threshold-select':
        dbc.Input(
            id='sb-threshold-select',
            type='number',
            min=0,
            max=0.05,
            step=0.0001,
            value=0.0005,
            disabled=True
        ),
    'sb-checklist-options':
        dbc.Checklist(
            options=[
                {'label': 'Visualize dot detection',
                 'value': 'vis_dot_det', 'disabled': True},
                {'label': 'Only decode dots in cells',
                 'value': 'decode_cells_only', 'disabled': True},
                {'label': 'All post-analyses', 'value': 'all_post', 'disabled': True},
                {'label': 'Nuclei labeled image', 'value': 'nuc_label', 'disabled': True},
            ],
            value=['vis_dot_det', 'decode_cells_only', 'all_post', 'nuc_label'],
            id='sb-checklist-options',
            switch=True
        ),
}

cm = ComponentManager(clear_components)

layout = html.Div([
    *[cm.component(k) for k in clear_components.keys()]
], style={'width': '500px'})
