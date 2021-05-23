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

import plotly.express as px
import plotly.graph_objects as go

from app import app
from lib.util import safe_imread
from .common import ComponentManager, data_clients

PAGENAME = 'dotdetection'
data_client = data_clients[PAGENAME]


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
                color=dots_select['z'].values))
        )

    return fig


clear_components = {

    'dd-analysis-select':
        dcc.Dropdown(
            id='dd-analysis-select',
            placeholder='Select an analysis'
        ),

    'dd-new-analysis-name': dcc.Input(type='text', id='dd-new-analysis-name'),
    'dd-submit-new-analysis-provider':
        dcc.ConfirmDialogProvider(
            html.Button('Submit new dot detection preview', n_clicks=0),
            id='dd-submit-new-analysis-provider',
            message='Confirm submission of new dot detection preview'
        ),
    'dd-new-analysis-text': html.Div(id='dd-new-analysis-text'),

    'dd-hyb-select':
        dcc.Dropdown(id='dd-hyb-select', placeholder='Select a hyb round'),
    'dd-position-select':
        dcc.Dropdown(id='dd-position-select', placeholder='Select a position'),
    'dd-swap-channels-slices':
        dcc.Checklist(
            id='dd-swap-channels-slices',
            options=[{'label': 'Swap channels and slices', 'value': 'swap'}],
            value=[]
        ),

    'dd-z-cap': html.B('Select Z slice'),
    'dd-chan-cap': html.B('Select a channel'),
    'dd-contrast-cap': html.B('Adjust contrast'),
    'dd-z-select': dcc.Slider(id='dd-z-select'),
    'dd-chan-select': dcc.Dropdown(id='dd-chan-select'),
    'dd-contrast-slider': dcc.RangeSlider(id='dd-contrast-slider'),

    'dd-fig': dcc.Graph(id='dd-fig')
}

component_groups = {
    'dataset-info': ['dd-analysis-select'],

    'new-analysis': ['dd-new-analysis-name',
                     'dd-submit-new-analysis-provider',
                     'dd-new-analysis-text'],

    'image-select': ['dd-hyb-select',
                     'dd-position-select',
                     'dd-swap-channels-slices'],

    'image-params': ['dd-z-cap',
                     'dd-z-select',
                     'dd-chan-cap',
                     'dd-chan-select',
                     'dd-contrast-cap',
                     'dd-contrast-slider']
}

cm = ComponentManager(clear_components, component_groups=component_groups)


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
    Input('dataset-select', 'value'),
    Input('user-select', 'value'),
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
    # Again test for None explicitly because z, channel, position or hyb might be 0
    if not is_open or any([v is None for v in
            (z, channel, contrast, position, hyb, dataset, user)]):
        return cm.component('dd-fig')

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

    return cm.component('dd-fig', figure=figure)


@app.callback(
    Output('dd-image-params-wrapper', 'is_open'),
    Output('dd-image-params-loader', 'children'),
    Input('dd-swap-channels-slices', 'value'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    Input('dataset-select', 'value'),
    Input('user-select', 'value')
)
def select_pos_hyb(swap, position, hyb, dataset, user):
    # Note we test for None rather than truthiness because a position or hyb of 0
    # evaluates to False when cast to bool, but is in fact a real value.
    is_open = not any([v is None for v in (position, hyb, dataset, user)])

    # Close the collapser and reset the components
    if not is_open:
        return False, cm.component_group('image-params', tolist=True)

    swap = 'swap' in swap

    imagefile = data_client.request(
        {'user': user, 'dataset': dataset, 'hyb': hyb, 'position': position},
        fields='hyb_fov'
    )

    try:
        image = safe_imread(imagefile['hyb_fov'][0])
        assert image.ndim == 4, 'Must have 4 dimensions'
    except (AssertionError, IndexError, RuntimeError) as e:
        print(e, type(e))
        return True, [
                   dbc.Alert(f'No image file for dataset {user}/{dataset} '
                             f'hyb {hyb} position {position} found!', color='warning')
               ] + cm.component_group('image-params',
                                      tolist=True,
                                      options=dict(disabled=True))

    if swap:
        z_ind = 0
        c_ind = 1
    else:
        z_ind = 1
        c_ind = 0

    z_range = list(range(image.shape[z_ind]))
    chan_range = list(range(image.shape[c_ind]))

    marks = {a // 256: str(a) for a in range(0, 10000, 500)}

    return True, [
        html.B('Select Z slice'),
        cm.component(
            'dd-z-select',
            min=-1,
            max=z_range[-1],
            step=1,
            value=0,
            marks={-1: 'Max'} | {z: str(z) for z in z_range}
        ),

        html.B('Select channel'),
        cm.component(
            'dd-chan-select',
            placeholder='Select channel',
            options=[{'label': str(c), 'value': str(c)} for c in chan_range],
            value='0'
        ),

        html.B('Adjust contrast'),
        html.Div([
            cm.component(
                'dd-contrast-slider',
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
    Output('dd-analysis-select', 'options'),
    Input('dataset-select', 'value'),
    State('user-select', 'value')
)
def select_dataset_analysis(dataset, user):
    if not dataset:
        return []

    analyses = data_client.datasets.query(
        'user==@user and dataset==@dataset')['analysis'].dropna().unique()

    return [{'label': '(new)', 'value': '__new__'}] +\
           [{'label': a, 'value': a} for a in analyses]


@app.callback(
    Output('dd-image-select-wrapper', 'children'),
    Input('dd-image-select-wrapper', 'is_open'),
    State('dataset-select', 'value'),
    State('user-select', 'value')
)
def display_image_selectors(is_open, dataset, user):
    if not is_open:
        return cm.component_group('image-select', tolist=True)

    print(user, dataset)

    hybs = data_client.datafiles.query(
        'user == @user and dataset == @dataset')['hyb'].dropna().unique()

    hybs_sorted = np.sort(hybs.astype(int)).astype(str)

    positions = data_client.datafiles.query(
        'user == @user and dataset == @dataset')['position'].dropna().unique()

    positions_sorted = np.sort(positions.astype(int)).astype(str)

    return cm.component_group(
        'image-select',
        tolist=True,
        options=
        {
            'dd-hyb-select':
                dict(options=[{'label': h, 'value': h} for h in hybs_sorted]),
            'dd-position-select':
                dict(options=[{'label': p, 'value': p} for p in positions_sorted])
        }
    )


@app.callback(
    [Output('dd-new-analysis-div', 'children'),
     Output('dd-image-select-wrapper', 'is_open'),
     Output('dd-analysis-select', 'value'),
     ],
    [Input('dataset-select', 'value'),
     Input('user-select', 'value')
     ]
)
def reset_dependents(dataset, user):
    # Always reset the new-analysis-div on user/dataset change
    new_analysis = cm.component_group('new-analysis', tolist=True)

    # Close the image-select wrapper if either is not defined
    image_select_open = dataset and user

    # Always reset the analysis value
    analysis_value = None

    return new_analysis, image_select_open, analysis_value


@app.callback(
    Output('dd-new-analysis-text', 'children'),
    Input('dd-submit-new-analysis-provider', 'submit_n_clicks'),
    State('dd-new-analysis-name', 'value'),
    State('user-select', 'value'),
    State('dataset-select', 'value'),
    State('dd-analysis-select', 'options')
)
def submit_new_analysis(
    confirm_n_clicks,
    new_analysis_name,
    user,
    dataset,
    analysis_options
):
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
                             f'analysis. Exception: ', color='error'),
                    html.Pre(new_analysis_future)
            ]

        return dbc.Alert(f'Success! New dot detection test will be at '
                         f'{user}/{dataset}/{new_analysis_future} '
                         f'in 5-10 minutes.', color='success')
    else:
        raise PreventUpdate


layout = html.Div([
    dbc.Alert('In this tab you can preview the results of dot detection'
                  ' using various parameter settings. Data is synced from the'
                  ' HPC analyses folder every 10 minutes.', color='info'),

    dbc.Row([

        dbc.Col([

            html.Div([

                *cm.component_group('dataset-info', tolist=True),

                html.Hr(),

                html.Div([
                    *cm.component_group('new-analysis', tolist=True)
                ], id='dd-new-analysis-div'),

                html.Hr(),

                dbc.Collapse([
                    *cm.component_group('image-select', tolist=True)
                ], is_open=False, id='dd-image-select-wrapper'),

            ], id='dd-dataset-select-div', style={'margin': '10px'}),

            html.Hr(),

            html.Div([

                dbc.Collapse([
                    dbc.Spinner([
                        *cm.component_group('image-params', tolist=True)
                    ], id='dd-image-params-loader')
                ], is_open=False, id='dd-image-params-wrapper'),

            ], id='dd-image-params-div', style={'margin': '10px'})

        ], style={'border-right': '1px solid gray'}, width=4),

        dbc.Col([
            dcc.Loading(cm.component('dd-fig'), id='dd-graph-wrapper')
        ], id='dd-fig-col')
    ])
])
