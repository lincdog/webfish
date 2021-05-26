import numpy as np
import pandas as pd
import io
import json
import re
from pathlib import Path

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import no_update

from app import app
from lib.util import sanitize, f2k
from .common import ComponentManager, data_clients, all_datasets


data_client = data_clients['dotdetection']

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


pipeline_stages = [
    'alignment',
    'dot detection',
    'segmentation',
    'decoding',
    'segmentation'
]
stage_ids = [
    'alignment',
    'dot-detection',
    'segmentation',
    'decoding',
    'segmentation-advanced'
]
stage_headers = [
    'Alignment Options',
    'Dot Detection Options',
    'Segmentation Options',
    'Decoding Options',
    'Advanced Segmentation Options',
]

clear_components = {
    # basic-metadata
    'sb-analysis-name':
        dbc.FormGroup([
            dbc.Label('Analysis Name', html_for='sb-analysis-name'),
            dbc.Input(
                id='sb-analysis-name',
                placeholder='Type a name for the analysis'
            )
        ]),
    'sb-position-select':
        dbc.FormGroup([
            dbc.Label('Position Select', html_for='sb-position-select'),
            dcc.Dropdown(
                id='sb-position-select',
                multi=True,
                placeholder='Select one or more positions (blank for all)'
            ),
            dbc.FormText('Select one or more positions, or leave blank to process all.')
        ]),

    # stage selection
    'sb-stage-select':
        dbc.Checklist(
            id='sb-stage-select',
            options=[
                {'label': 'Alignment', 'value': 'alignment'},
                {'label': 'Dot Detection', 'value': 'dot detection'},
                {'label': 'Segmentation', 'value': 'segmentation'},
                {'label': 'Decoding', 'value': 'decoding'}
            ],
            value=['alignment', 'dot detection', 'segmentation', 'decoding'],
            switch=True
        ),

    # alignment
    'sb-alignment-select':
        dbc.FormGroup([
            dbc.Label('Alignment Algorithm', html_for='sb-alignment-select'),
            dbc.Select(
                id='sb-alignment-select',
                value='mean squares 2d',
                options=[{'label': 'Mean Squares 2D', 'value': 'mean squares 2d'}],
                disabled=True
            ),
        ]),

    # dot-detection
    'sb-dot-detection-select':
        dbc.FormGroup([
            dbc.Label('Dot Detection Algorithm', html_for='sb-dot-detection-select'),
            dbc.Select(
                id='sb-dot-detection-select',
                options=[
                    {'label': 'Biggest Jump 3D', 'value': 'biggest jump 3d'},
                    {'label': 'ADCG 2D', 'value': 'adcg 2d'}
                ],
                value='biggest jump 3d',
                disabled=True
            ),
        ]),
    'sb-strictness-select':
        dbc.FormGroup([
            dbc.Label('Strictness parameter', html_for='sb-strictness-select'),
            dcc.Slider(
                id='sb-strictness-select',
                min=-15,
                max=15,
                step=1,
                value=2,
                marks={i: str(i) for i in range(-15, 16, 2)}
            ),
            dbc.FormText('Higher strictness sets a '
                         'higher minimum intensity threshold for dot detection.')
        ]),
    'sb-threshold-select':
        dbc.FormGroup([
            dbc.Label('Laplacian Threshold (advanced)', html_for='sb-threshold-select'),
            dbc.Input(
                id='sb-threshold-select',
                type='number',
                min=0,
                max=0.05,
                step=0.0001,
                value=0.0005,
                disabled=True
            ),
            dbc.FormText('Set a threshold of the LoG filter. Default is usually fine.')
        ]),
    'sb-dotdetection-checklist':
        dbc.Checklist(
            options=[
                {'label': 'Visualize dot detection',
                 'value': 'visualize dot detection', 'disabled': True},
            ],
            value=['visualize dot detection'],
            id='sb-dotdetection-checklist',
            switch=True
        ),

    # segmentation
    'sb-segmentation-select':
        dbc.FormGroup([
            dbc.Label('Segmentation Algorithm', html_for='sb-segmentation-select'),
            dbc.RadioItems(
                id='sb-segmentation-select',
                options=[
                    {'label': 'Cellpose', 'value': 'cellpose'},
                ],
                value='cellpose'
            ),
        ]),
    'sb-segmentation-checklist':
        dbc.Checklist(
            id='sb-segmentation-checklist',
            options=[
                {'label': 'Only decode dots in cells',
                 'value': 'only decode dots in cells'},
                {'label': 'Run all post-analyses',
                 'value': 'all post analyses', 'disabled': True},
                {'label': 'Segment nuclei',
                 'value': 'nuclei labeled image'},
                {'label': 'Segment cytoplasm',
                 'value': 'cyto labeled image'},
            ],
            value=['only decode dots in cells',
                   'all post analyses',
                   'nuclei labeled image'],
            switch=True
        ),
    # segmentation-advanced
    'sb-edge-deletion':
        dbc.FormGroup([
            dbc.Label('Edge Deletion', html_for='sb-edge-deletion'),
            dbc.Input(
                id='sb-edge-deletion',
                type='number',
                min=0,
                max=20,
                step=1,
                value=8
            ),
            dbc.FormText('Set the number of pixels to be deleted between neighboring '
                         'cells in the labeled image.')
        ]),
    'sb-nuclei-distance':
        dbc.FormGroup([
            dbc.Label('Nuclei Distance', html_for='sb-nuclei-distance'),
            dbc.Input(
                id='sb-nuclei-distance',
                type='number',
                min=0,
                max=10,
                step=1,
                value=2
            ),
        ]),
    'sb-cyto-channel':
        dbc.FormGroup([
            dbc.Label('Cytoplasm Channel', html_for='sb-cyto-channel'),
            dbc.RadioItems(
                id='sb-cyto-channel',
                options=[{'label': '1', 'value': '1'},
                         {'label': '2', 'value': '2'},
                         {'label': '3', 'value': '3'}
                         ],
                value='3',
                inline=True
            ),
            dbc.FormText('Select which channel to use for cytoplasm segmentation')
        ]),
    'sb-nuclei-radius':
        dbc.FormGroup([
            dbc.Label('Select Nuclear Radius', html_for='sb-nuclei-radius'),
            dbc.Input(
                id='sb-nuclei-radius',
                type='number',
                min=0,
                max=100,
                step=1,
                value=10
            ),
        ]),
    'sb-cell-prob-threshold':
        dbc.FormGroup([
            dbc.Label('Set Cell Probability Threshold', html_for='sb-cell-prob-threshold'),
            dcc.Slider(
                id='sb-cell-prob-threshold',
                min=-6,
                max=6,
                step=1,
                value=-4,
                marks={i: str(i) for i in range(-6, 7, 1)}
            ),
        ]),
    'sb-flow-threshold':
        dbc.FormGroup([
            dbc.Label('Set Flow Threshold Parameter', html_for='sb-flow-threshold'),
            dcc.Slider(
                id='sb-flow-threshold',
                min=0,
                max=1,
                step=0.01,
                value=0.8,
                marks={i: f'{i:0.2f}' for i in np.linspace(0, 1, 11)}
            )
        ]),

    # Decoding
    'sb-decoding-select':
        dbc.FormGroup([
            dbc.Label('Select Decoding Algorithm', html_for='sb-decoding-select'),
            dbc.RadioItems(
                id='sb-decoding-select',
                options=[
                    {'label': 'Across channels', 'value': 'across'},
                    {'label': 'Individual channel(s)', 'value': 'individual'}
                ],
                value='individual',
                inline=True
            ),
            dbc.FormText('If "Individual" is selected, also select which channels, below.')
        ]),
    'sb-individual-channel-select':
        dbc.FormGroup([
            dbc.Label('Select Channels for Individual Decoding',
                      html_for='sb-individual-channel-select'),
            dbc.Checklist(
                options=[
                    {'label': '1', 'value': '1'},
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'}
                ],
                id='sb-individual-channel-select',
                inline=True
            ),
            dbc.FormText('This only has an effect if "Individual" is selected above.')
        ]),

}

component_groups = {
    'basic-metadata': ['sb-analysis-name',
                       'sb-position-select',
                       'sb-stage-select'],

    'alignment': ['sb-alignment-select'],

    'dot-detection': ['sb-dot-detection-select',
                      'sb-strictness-select',
                      'sb-threshold-select',
                      'sb-dotdetection-checklist'],

    'segmentation': ['sb-segmentation-select',
                     'sb-segmentation-checklist'],

    'segmentation-advanced': ['sb-edge-deletion',
                              'sb-nuclei-distance',
                              'sb-cyto-channel',
                              'sb-nuclei-radius',
                              'sb-cell-prob-threshold',
                              'sb-flow-threshold'],

    'decoding': ['sb-decoding-select',
                 'sb-individual-channel-select']

}

def _position_process(positions):
    if not positions:
        return {'positions': ''}

    return {'positions': ','.join([str(p) for p in positions])}

def _checklist_process(checked):
    return {k: "true" for k in checked}

def _decoding_channel_process(arg):
    if arg == 'across':
        return {'decoding': 'across'}
    elif arg == 'individual':
        return {}
    elif isinstance(arg, list):
        return {'decoding': {
            'individual': [str(a) for a in arg]
        }}
    else:
        return {}

id_to_json_key = {
    'user-select': 'personal',
    'dataset-select': 'experiment_name',

    'sb-position-select': _position_process,

    'sb-alignment-select': 'alignment',

    'sb-dot-detection-select': 'dot detection',
    'sb-strictness-select': 'strictness',
    'sb-threshold-select': 'threshold',
    'sb-dot-detection-checklist': _checklist_process,

    'sb-segmentation-select': 'segmentation',
    'sb-segmentation-checklist': _checklist_process,
    'sb-edge-deletion': 'edge deletion',
    'sb-nuclei-distance': 'distance between nuclei',
    'sb-cyto-channel': 'cyto channel number',
    'sb-nuclei-radius': 'nuclei radius',
    'sb-cell-prob-threshold': 'cell_prob_threshold',
    'sb-flow-threshold': 'flow_threshold',

    'sb-decoding-select': _decoding_channel_process,
    'sb-individual-channel-select': _decoding_channel_process
}


def form_to_json_output(form_status):
    out = {
        "clusters": {
            "ntasks": "1",
            "mem-per-cpu": "10G",
            "email": "nrezaee@caltech.edu"
        }
    }

    analysis_name = ''

    for k, v in form_status.items():

        if k == 'sb-analysis-name' and v:
            analysis_name = sanitize(v, delimiter_allowed=False)
        elif k in id_to_json_key.keys():
            prekey = id_to_json_key[k]

            if callable(prekey):
                update = prekey(v)
            else:
                update = {prekey: str(v)}

            for uk, uv in update.items():
                if uk not in out.keys():
                    out[uk] = uv

    return analysis_name, out


cm = ComponentManager(clear_components, component_groups=component_groups)


@app.callback(
    Output('sb-position-select', 'options'),
    Input('user-select', 'value'),
    Input('dataset-select', 'value')
)
def select_user_dataset(user, dataset):
    if not user or not dataset:
        raise PreventUpdate

    query = 'user==@user and dataset==@dataset and source_key=="hyb_fov"'

    positions = np.sort(data_client.datafiles.query(
        query)['position'].unique().astype(int))

    return [{'label': p, 'value': p} for p in list(positions)]

    #analyses = data_client.datafiles.query(
    #    'user==@user and dataset==@dataset')['analysis'].unique()


@app.callback(
    [Output(f'sb-{stage}-wrapper', 'open')
     for stage in stage_ids],
    [Output(f'sb-{stage}-wrapper', 'children')
     for stage in stage_ids],
    Input('sb-stage-select', 'value')
)
def select_pipeline_stages(stages):
    if stages is None:
        raise PreventUpdate

    if 'segmentation' in stages:
        stages.append('segmentation')

    selected_ids = [
        stage_id
        for stage, stage_id
        in zip(pipeline_stages, stage_ids)
        if stage in stages
    ]

    print(selected_ids)

    open_res = [i in selected_ids for i in stage_ids]
    open_res[-1] = False  # advanced segmentation

    children_res = [
        [html.Summary(header),
         *cm.component_group(group_id, tolist=True, options=dict(disabled=open))]
        for header, group_id, open in zip(stage_headers, stage_ids, open_res)
    ]

    return tuple(open_res + children_res)


@app.callback(
    Output('sb-submission-json', 'children'),
    Input('sb-submit-button', 'n_clicks'),
    State('user-select', 'value'),
    State('dataset-select', 'value'),
    State('sb-stage-select', 'value'),
    [State(comp, 'value') for comp in cm.clear_components.keys()
     if comp != 'sb-stage-select']
)
def submit_new_analysis(n_clicks, user, dataset, stage_select, *values):
    if not n_clicks:
        raise PreventUpdate

    upload = True

    relevant_comps = [c for c in cm.clear_components.keys() if c != 'sb-stage-select']

    status = {'user-select': user, 'dataset-select': dataset}
    status.update({c: v for c, v in zip(relevant_comps, values)})

    analysis_name, submission = form_to_json_output(status)

    print(stage_select, submission)
    for stage in set(pipeline_stages):
        if stage not in stage_select:
            submission.pop(stage, None)

    analyses = all_datasets.query(
        'user==@user and dataset==@dataset')['analysis'].unique()

    alerts = []

    if not all((analysis_name, user, dataset)):
        upload = False
        alerts.append(dbc.Alert('Please choose a user and dataset, and specify'
                                ' a new analysis name.', color='danger'))

    if analysis_name in analyses:
        upload = False
        alerts.append(
            dbc.Alert(dcc.Markdown(f'An analysis named `{analysis_name}` '
                      f'already exists for the chosen dataset.'), color='danger'))

    submission_bytes = io.BytesIO(json.dumps(submission).encode())

    if upload:
        try:
            data_client.client.client.upload_fileobj(
                submission_bytes,
                Bucket=data_client.bucket_name,
                Key=f2k(Path('json_analyses', analysis_name + '.json'))
            )
            alerts.append(dbc.Alert(dcc.Markdown(f'Successfully uploaded new analysis `{analysis_name}`!'
                                    f' The full path will be: `{user}/{dataset}/{analysis_name}`'),
                                    color='success'))
        except Exception as e:
            alerts.append(dbc.Alert('Error uploading JSON to S3: ', e, color='danger'))
    else:
        alerts.append(dbc.Alert('Did not upload JSON to S3 due to errors', color='danger'))

    return [
        *alerts,
        dcc.Markdown(f'New analysis name: `{analysis_name}`'),
        html.Summary('Generated JSON'),
        html.Pre(json.dumps(
            submission, indent=2)),
    ]


@app.callback(
    [Output('user-select', 'value'),
     Output('sb-col-1', 'children'),
     Output('sb-col-2', 'children')],
    Input('sb-reset-button', 'n_clicks')
)
def reset_to_defaults(n_clicks):
    if n_clicks:
        return None, col1_clear, col2_clear
    else:
        raise PreventUpdate


col1_clear = [
    html.Details(
        [html.Summary('Basic information')] +
        cm.component_group('basic-metadata', tolist=True),
        open=True, id='sb-basic-metadata-wrapper'
    ),
    html.Hr(),
    html.Details(
        [html.Summary('Alignment options')] +
        cm.component_group('alignment', tolist=True),
        open=True, id='sb-alignment-wrapper'
    ),
    html.Hr(),
    html.Details(
        [html.Summary('Dot detection options')] +
        cm.component_group('dot-detection', tolist=True),
        open=True, id='sb-dot-detection-wrapper'
    ),
]

col2_clear = [
    html.Details(
        [html.Summary('Segmentation options')] +
        cm.component_group('segmentation', tolist=True),
        open=True, id='sb-segmentation-wrapper'
    ),
    html.Hr(),
    html.Details(
        [html.Summary('Advanced segmentation options')] +
        cm.component_group('segmentation-advanced', tolist=True),
        open=False, id='sb-segmentation-advanced-wrapper'
    ),
    html.Hr(),
    html.Details(
        [html.Summary('Decoding options')] +
        cm.component_group('decoding', tolist=True),
        open=True, id='sb-decoding-wrapper'
    ),
    dbc.Card([
        dbc.CardHeader('Submission'),
        dbc.CardBody([
            dbc.Button('Submit', id='sb-submit-button', color='primary', size='lg'),
            dbc.Button('Reset to defaults', id='sb-reset-button', color='warning', size='lg'),
            html.Details(html.Summary('Generated JSON'), id='sb-submission-json', open=True),
        ]),
    ]),
]

layout = [
    dbc.Col(col1_clear, id='sb-col-1', width=4),
    dbc.Col(col2_clear, id='sb-col-2', width=4)
]

