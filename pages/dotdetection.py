import tifffile as tif
import numpy as np
import pandas as pd
import io
import json
import re
import time

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go

from app import app, config, s3_client
from lib.util import safe_imread
from lib import cloud

data_client = cloud.DataClient(
    config=config,
    s3_client=s3_client,
    pagename='dotdetection'
)
data_client.sync_with_s3()


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


def gen_image_figure(
    imfile,
    dots_csv=None,
    swap=False,
    hyb='0',
    z_slice='0',
    channel='0',
    contrast_minmax=(0, 2000)
):

    if len(imfile) > 0:
        image = safe_imread(imfile[0])
    else:
        return {}

    print(f'hyb {hyb} z_slice {z_slice} channel {channel}')
    print(image.shape)

    hyb = int(hyb)
    hyb_q = hyb
    # 'z' column in locations.csv starts at 0
    z_slice = int(z_slice)
    z_slice_q = z_slice
    # 'ch' column in locations.csv starts at 1
    channel = int(channel)
    channel_q = channel + 1

    if z_slice >= 0:
        if swap:
            img_select = image[z_slice, channel]
        else:
            img_select = image[channel, z_slice]

        dots_query = 'hyb == @hyb_q and ch == @channel_q and z == @z_slice_q'
    else:
        if swap:
            img_select = np.max(image[:, channel], axis=0)
        else:
            img_select = np.max(image[channel], axis=0)
        dots_query = 'hyb == @hyb_q and ch == @channel_q'

    fig = px.imshow(
        img_select,
        zmin=contrast_minmax[0],
        zmax=contrast_minmax[1],
        width=1000,
        height=1000,
        binary_string=True
    )

    if dots_csv:
        dots_select = pd.read_csv(
            dots_csv[0],
            #dtype={'ch': int, 'z': int, 'hyb': int}
        ).query(dots_query)

        print(dots_query, dots_select.info())

        fig.add_trace(go.Scatter(
            x=dots_select['x'].values,
            y=dots_select['y'].values,
            mode='markers',
            marker_symbol='cross',
            marker=dict(
                maxdisplayed=1000,
                size=10,
                color=dots_select['int'].values))
        )

    return fig


@app.callback(
    Output('dd-graph-wrapper', 'children'),
    Input('dd-image-params-wrapper', 'is_open'),
    Input('dd-swap-channels-slices', 'value'),
    Input('dd-z-select', 'value'),
    Input('dd-chan-select', 'value'),
    Input('dd-contrast-slider', 'value'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    Input('dd-analysis-select', 'value'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value'),
)
def update_image_params(
    is_open,
    swap,
    z,
    channel,
    contrast,
    position,
    hyb,
    analysis,
    dataset,
    user
):
    if not is_open:
        raise PreventUpdate

    if any([v is None for v in
            (z, channel, contrast, position, hyb, dataset, user)]):
        return {}

    swap = 'swap' in swap

    hyb_fov = data_client.request(
        {'user': user, 'dataset': dataset, 'position': position, 'hyb': hyb},
        fields='hyb_fov'
    )['hyb_fov']

    if analysis:
        dot_locations = data_client.request(
            {'user': user, 'dataset': dataset, 'position': position, 'analysis': analysis},
            fields='dot_locations'
        )['dot_locations']
    else:
        dot_locations = None

    print(f'hyb_fov: {hyb_fov}')
    print(f'dot locations: {dot_locations}')

    figure = gen_image_figure(hyb_fov, dot_locations, swap, hyb, z, channel, contrast)

    return dcc.Graph(figure=figure, id='dd-fig')


@app.callback(
    Output('dd-image-params-wrapper', 'children'),
    Input('dd-image-params-wrapper', 'is_open'),
    Input('dd-swap-channels-slices', 'value'),
    State('dd-position-select', 'value'),
    State('dd-hyb-select', 'value'),
    State('dd-dataset-select', 'value'),
    State('dd-user-select', 'value')
)
def display_image_param_selectors(is_open, swap, position, hyb, dataset, user):
    if not is_open:
        raise PreventUpdate

    swap = 'swap' in swap

    imagefile = data_client.request(
        {'user': user, 'dataset': dataset, 'hyb': hyb, 'position': position},
        fields='hyb_fov'
    )

    try:
        image = safe_imread(imagefile['hyb_fov'][0])
        assert image.ndim == 4, 'Must have 4 dimensions'
    except (AssertionError, IndexError):
        return [
            dbc.Alert(f'No image file for dataset {user}/{dataset} '
                      f'hyb {hyb} position {position} found!', color='warning'),
            dcc.Slider(id='dd-z-select', disabled=True),
            dcc.Dropdown(id='dd-chan-select', disabled=True),
            dcc.Slider(id='dd-contrast-slider', disabled=True)
            ]

    if swap:
        z_ind = 0
        c_ind = 1
    else:
        z_ind = 1
        c_ind = 0

    z_range = list(range(image.shape[z_ind]))
    chan_range = list(range(image.shape[c_ind]))

    marks = {a // 256: str(a) for a in range(0, 10000, 500)}

    return [
        html.Hr(),
        html.B('Select Z slice'),
        dcc.Slider(
            id='dd-z-select',
            min=-1,
            max=z_range[-1],
            step=1,
            value=0,
            marks={-1: 'Max'} | {z: str(z) for z in z_range}
        ),
        html.B('Select channel'),
        dcc.Dropdown(
            id='dd-chan-select',
            placeholder='Select channel',
            options=[{'label': str(c), 'value': str(c)} for c in chan_range],
            value='0'
        ),
        html.B('Adjust contrast'),
        html.Div([
            dcc.RangeSlider(
                id='dd-contrast-slider',
                min=0,
                max=10000 // 256,
                step=1,
                marks=marks,
                value=[0, 10],
                allowCross=False
            ),
        ], id='dd-contrast-div'),
    ]


@app.callback(
    Output('dd-image-params-wrapper', 'is_open'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value')
)
def select_pos_hyb(position, hyb, dataset, user):
    if any([v is None for v in (position, hyb, dataset, user)]):
        return False

    return True


@app.callback(
    Output('dd-analysis-select', 'options'),
    Input('dd-dataset-select', 'value'),
    State('dd-user-select', 'value')
)
def select_dataset_analysis(dataset, user):
    if not dataset:
        return []

    analyses = data_client.datasets.loc[(user, dataset)].index.unique(level=0).dropna()

    print(analyses)

    return [{'label': '(new)', 'value': '__new__'}] +\
           [{'label': a, 'value': a} for a in analyses]


@app.callback(
    Output('dd-analysis-select', 'value'),
    Input('dd-user-select', 'value')
)
def clear_analysis_select(user):
    return None


@app.callback(
    Output('dd-image-select-wrapper', 'children'),
    Input('dd-image-select-wrapper', 'is_open'),
    State('dd-dataset-select', 'value'),
    State('dd-user-select', 'value')
)
def display_image_selectors(is_open, dataset, user):
    if not is_open:
        raise PreventUpdate

    hybs = data_client.datafiles.query(
        'user == @user and dataset == @dataset')['hyb'].dropna().unique()

    positions = data_client.datafiles.query(
        'user == @user and dataset == @dataset')['position'].dropna().unique()

    return [
        dcc.Dropdown(
            id='dd-hyb-select',
            options=[{'label': h, 'value': h} for h in sorted(hybs)],
            placeholder='Select hyb',
            clearable=False
        ),
        dcc.Dropdown(
            id='dd-position-select',
            options=[{'label': p, 'value': p} for p in sorted(positions)],
            placeholder='Select position',
            clearable=False
        ),
        dcc.Checklist(
            id='dd-swap-channels-slices',
            options=[{'label': 'Swap channels and slices', 'value': 'swap'}],
            value=[]
        ),
    ]


@app.callback(
    Output('dd-image-select-wrapper', 'is_open'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value')
)
def select_dataset_image(dataset, user):
    if not dataset:
        return False

    return True


@app.callback(
    Output('dd-dataset-select', 'value'),
    Input('dd-dataset-select', 'options')
)
def clear_dataset_select(opts):
    return None


@app.callback(
    Output('dd-dataset-select', 'options'),
    Input('dd-user-select', 'value')
)
def select_user(user):
    if not user:
        return []

    datasets = data_client.datasets.loc[user].index.unique(level=0)

    return [{'label': d, 'value': d} for d in sorted(datasets)]


@app.callback(
    Output('dd-new-analysis-div', 'children'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value')
)
def reset_new_analysis_div(dataset, user):
    return [
        dcc.Input(type='text', id='dd-new-analysis-name', value=None),
        dcc.ConfirmDialogProvider(
            html.Button('Submit new analysis'),
            id='dd-submit-new-analysis-provider',
            message='Confirm submission of new dot detection preview'
        ),
        html.Div(id='dd-new-analysis-text')
    ]


@app.callback(
    Output('dd-new-analysis-text', 'children'),
    Input('dd-submit-new-analysis-provider', 'submit_n_clicks'),
    State('dd-new-analysis-name', 'value'),
    State('dd-user-select', 'value'),
    State('dd-dataset-select', 'value'),
    State('dd-analysis-select', 'options')
)
def submit_new_analysis(
    confirm_n_clicks,
    new_analysis_name,
    user,
    dataset,
    analysis_options
):
    print(confirm_n_clicks, new_analysis_name, user, dataset, analysis_options)
    if not all((new_analysis_name, user, dataset, analysis_options)):
        raise PreventUpdate

    analyses = [o['value'] for o in analysis_options]

    if new_analysis_name in analyses:
        return dbc.Alert(f'Analysis {new_analysis_name} already exists. '
                         f'Please choose a unique name.', color='error')

    if confirm_n_clicks == 1:
        new_analysis_future = put_analysis_request(
            user,
            dataset,
            new_analysis_name
        )

        if isinstance(new_analysis_future, Exception):
            return [dbc.Alert(f'Failure: failed to submit request for new '
                             f'analysis. Exception: '),
                    html.Pre(new_analysis_future)
            ]

        return dbc.Alert(f'Success! New dot detection test will be at '
                         f'{user}/{dataset}/{new_analysis_future} '
                         f'in 5-10 minutes.')
    else:
        raise PreventUpdate


@app.callback(
    Output('dd-s3-sync-div', 'children'),
    Input('dd-s3-sync-button', 'children'),
    State('dd-s3-sync-button', 'n_clicks')
)
def finish_client_s3_sync(contents, n_clicks):
    if n_clicks != 1:
        raise PreventUpdate

    if 'Syncing...' not in contents:
        raise PreventUpdate

    data_client.sync_with_s3()

    return dbc.Button(
                'Sync with S3',
                id='dd-s3-sync-button',
                color='primary',
                className='mr-1',
                n_clicks=0
            )


@app.callback(
    Output('dd-s3-sync-button', 'children'),
    Input('dd-s3-sync-button', 'n_clicks')
)
def init_client_s3_sync(n_clicks):
    if n_clicks == 1:
        return [dbc.Spinner(size='sm'), 'Syncing...']
    else:
        raise PreventUpdate


layout = html.Div([
    dbc.Alert('In this tab you can preview the results of dot detection'
                  ' using various parameter settings. Data is synced from the'
                  ' HPC analyses folder every 10 minutes.', color='info'),
    html.Div(dbc.Button(
        'Sync with S3',
        id='dd-s3-sync-button',
        n_clicks=0,
        color='primary'
    ), id='dd-s3-sync-div'),
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id='dd-user-select',
                    options=[{'label': u, 'value': u} for u in data_client.datasets.index.unique(level=0)],
                    placeholder='Select a user'
                ),
                dcc.Dropdown(id='dd-dataset-select', placeholder='Select a dataset'),
                dcc.Dropdown(id='dd-analysis-select', placeholder='Select an analysis'),
                html.Hr(),
                html.Div([
                    dcc.Input(type='text', id='dd-new-analysis-name'),
                    dcc.ConfirmDialogProvider(
                        dbc.Button('Submit new analysis', color='secondary'),
                        id='dd-submit-new-analysis-provider',
                        message='Confirm submission of new dot detection preview'
                    ),
                    html.Div(id='dd-new-analysis-text')
                ], id='dd-new-analysis-div'),
                html.Hr(),
                dbc.Collapse([
                    dcc.Dropdown(id='dd-hyb-select'),
                    dcc.Dropdown(id='dd-position-select'),
                    dcc.Checklist(
                        id='dd-swap-channels-slices',
                        options=[{'label': 'Swap channels and slices', 'value': 'swap'}],
                        value=[]
                    ),
                ], is_open=False, id='dd-image-select-wrapper'),
            ], id='dd-dataset-select-div', style={'margin': '10px'}),
            html.Hr(),
            html.Div([
                dbc.Collapse([
                    dcc.Slider(id='dd-z-select'),
                    dcc.Dropdown(id='dd-chan-select'),
                    dcc.Slider(id='dd-contrast-slider')
                ], is_open=False, id='dd-image-params-wrapper'),
            ], id='dd-image-params-div', style={'margin': '10px'})

        ], style={'border-right': '1px solid gray'}, width=4),

        dbc.Col([
            dcc.Loading(dcc.Graph(id='dd-fig'), id='dd-graph-wrapper')
        ], id='dd-fig-col')
    ])
])
