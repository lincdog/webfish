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
from lib import cloud

data_client = cloud.DataClient(
    config=config,
    s3_client=s3_client,
    pagename='overview'
)
data_client.sync_with_s3()