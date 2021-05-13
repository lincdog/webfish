import tifffile as tif
import numpy as np
import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_slicer import VolumeSlicer

import plotly.express as px
import plotly.graph_objects as go

from app import app, config, s3_client
from lib import cloud
from lib.util import ImageMeta

data_client = cloud.DataClient(
    config=config,
    s3_client=s3_client,
    pagename='dotdetection'
)
data_client.sync_with_s3()


def gen_image_figure(
    imfile,
    dots_csv=None,
    hyb='0',
    z_slice='0',
    channel='0',
    contrast_minmax=(0, 2000)
):
    # FIXME: Seems that some images come in C, Z, X, Y order rather than the assumed
    #   Z, C, X, Y order! May need ImageMeta-type reading to ensure we get it right

    if len(imfile) > 0:
        image = ImageMeta(imfile[0]).asarray()
    else:
        return {}

    print(f'hyb {hyb} z_slice {z_slice} channel {channel}')

    hyb = int(hyb)
    hyb_q = hyb
    # 'z' column in locations.csv starts at 0
    z_slice = int(z_slice)
    z_slice_q = z_slice
    # 'ch' column in locations.csv starts at 1
    channel = int(channel)
    channel_q = channel + 1

    if z_slice >= 0:
        img_select = image[channel, z_slice]
        dots_query = 'hyb == @hyb_q and ch == @channel_q and z == @z_slice_q'
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
    Input('dd-z-select', 'value'),
    Input('dd-chan-select', 'value'),
    Input('dd-contrast-slider', 'value'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    State('dd-analysis-select', 'value'),
    State('dd-dataset-select', 'value'),
    State('dd-user-select', 'value'),
)
def update_image_params(
    z,
    channel,
    contrast,
    position,
    hyb,
    analysis,
    dataset,
    user
):
    if any([v is None for v in
            (z, channel, contrast, position, hyb, dataset, user)]):
        return {}

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

    figure = gen_image_figure(hyb_fov, dot_locations, hyb, z, channel, contrast)

    return dcc.Graph(figure=figure, id='dd-fig')


@app.callback(
    Output('dd-image-params-wrapper', 'children'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value')
)
def select_pos_hyb(position, hyb, dataset, user):
    if any([v is None for v in (position, hyb, dataset, user)]):
        return []

    imagefile = data_client.request(
        {'user': user, 'dataset': dataset, 'hyb': hyb, 'position': position},
        fields='hyb_fov'
    )

    if not imagefile['hyb_fov']:
        return html.H2(f'No image file for dataset {user}/{dataset} '
                       f'hyb {hyb} position {position} found!')

    image = ImageMeta(imagefile['hyb_fov'][0])

    print(imagefile['hyb_fov'], image.shape)

    z_range = range(image.shape[1])
    chan_range = range(image.shape[0])

    marks = {a//256: str(a) for a in range(0, 10000, 500)}

    return [
        #dcc.Dropdown(
        #    id='dd-z-select',
        #    options=[{'label': 'Max', 'value': '-1'}] +
        #            [{'label': str(z), 'value': str(z)} for z in z_range],
        #    placeholder='Select a Z slice',
        #    clearable=False
        #),
        dcc.Slider(
            id='dd-z-select',
            min=-1,
            max=image.shape[1],
            step=1,
            value=0,
            marks={-1: 'Max'} | {z: str(z) for z in z_range}
        ),
        dcc.Dropdown(
            id='dd-chan-select',
            placeholder='Select channel',
            options=[{'label': str(c), 'value': str(c)} for c in chan_range]
        ),
        html.Div([
            dcc.RangeSlider(
                id='dd-contrast-slider',
                min=0,
                max=10000//256,
                step=1,
                marks=marks,
                value=[0, 10],
                allowCross=False
            ),
        ], id='dd-contrast-div'),
    ]


@app.callback(
    Output('dd-analysis-select-wrapper', 'children'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value')
)
def select_dataset_analysis(dataset, user):
    if not dataset:
        return []

    analyses = data_client.datasets.loc[(user, dataset)].index.unique(level=0).dropna()

    print(analyses)

    return [
        dcc.Dropdown(
            id='dd-analysis-select',
            options=[{'label': '(new)', 'value': '__new__'}] +
                    [{'label': a, 'value': a} for a in analyses],
            placeholder='Select an analysis'
        )
    ]


@app.callback(
    Output('dd-image-select-wrapper', 'children'),
    Input('dd-dataset-select', 'value'),
    Input('dd-user-select', 'value')
)
def select_dataset_image(dataset, user):
    if not dataset:
        return []

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
        )
    ]


@app.callback(
    Output('dd-dataset-select-wrapper', 'children'),
    Input('dd-user-select', 'value')
)
def select_user(user):
    if not user:
        raise PreventUpdate

    datasets = data_client.datasets.loc[user].index.unique(level=0)

    return [
        dcc.Dropdown(
            id='dd-dataset-select',
            options=[{'label': d, 'value': d} for d in sorted(datasets)],
            placeholder='Select a dataset name'
        )
    ]


layout = html.Div([
    dbc.Alert('In this tab you can preview the results of dot detection'
                  ' using various parameter settings. Data is synced from the'
                  ' HPC analyses folder every 2 minutes.', color='info'),
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id='dd-user-select',
                    options=[{'label': u, 'value': u} for u in data_client.datasets.index.unique(level=0)],
                    placeholder='Select a user'
                ),
                dcc.Loading(id='dd-dataset-select-wrapper'),
                dcc.Loading(id='dd-analysis-select-wrapper'),
                dcc.Loading([
                    #dcc.Dropdown(id='dd-hyb-select'),
                    #dcc.Dropdown(id='dd-position-select')
                ], id='dd-image-select-wrapper'),
            ], id='dd-dataset-select-div', style={'margin': '10px'}),
            html.Hr(),
            html.Div([
                dcc.Loading(id='dd-image-params-wrapper')
            ], id='dd-image-params-div', style={'margin': '10px'})

        ], style={'border-right': '1px solid gray'}, width=4),

        dbc.Col([
            dcc.Loading(dcc.Graph(id='dd-fig'), id='dd-graph-wrapper')
        ], id='dd-fig-col')
    ])
])
