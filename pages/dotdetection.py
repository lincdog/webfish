import json
import tifffile as tif
import numpy as np
import pandas as pd
from pathlib import Path

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go

from app import config, s3_client
from lib import cloud

data_client = cloud.DataClient(
    config=config,
    s3_client=s3_client,
    pagename='dotdetection'
)
data_client.sync_with_s3()

dataset_root = Path('webfish_data', 'dotdetection', 'nrezaee')
hyb = 0
pos = 0
img_name = Path(dataset_root, 'raw', '2020-08-08-takei', f'HybCycle_{hyb}', f'compressed_MMStack_Pos{pos}.ome.tif')
dots_name = Path(dataset_root, '2020-08-08-takei', f'takei_strict_6/MMStack_Pos{pos}/Dot_Locations/locations.csv')

img = tif.imread(img_name) # Shape should be (13, 4, 2048, 2048) - ZCYX
dots_df = pd.read_csv(dots_name, dtype={'hyb': int, 'ch': int, 'z': int}).query('hyb == @hyb')
print(img.shape)
channel = 2
z = 5

fig = px.imshow(
    np.max(img[:, 0], axis=0),
    zmin=0,
    zmax=10,
    width=1000,
    height=1000,
    binary_string=True)

dots_select = dots_df.query('ch == 1')
print(len(dots_select))

fig.add_trace(go.Scatter(
    x=dots_select['x'].values-1,
    y=dots_select['y'].values-1,
    mode='markers',
    marker_symbol='cross',
    marker=dict(maxdisplayed=1000,
                size=10,
                color=dots_select['int'].values))
)

layout = html.Div([
    dbc.Alert('In this tab you can preview the results of dot detection'
                  ' using various parameter settings. Data is synced from the'
                  ' HPC analyses folder every 2 minutes.', color='info'),
    dbc.Row([
        dbc.Col([

        ], style={'border-right': '1px solid gray'}, width=4),
        dbc.Col([
            dcc.Graph(figure=fig)
        ])
    ])
])
