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
from flask_caching import Cache

from app import app
from util import gen_mesh, gen_pcd_df, mesh_from_json, populate_mesh, populate_genes
from cloud import s3_connect, grab_bucket, download_s3_folder

layout = dbc.Row([
    dbc.Col([
        html.Div(html.H1('hey'))
    ])
])