import dateutil
import logging
import re
import pandas as pd
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from datetime import datetime
from pathlib import PurePath, Path
from lib.util import find_matching_files
from app import app
from pages.common import data_client

logger = logging.getLogger('webfish.' + __name__)


def parse_line(line):
    # Currently the format, which is similar to the webfish format
    # defined in index.py, is:
    # <full/path/to/analysis/current/position> : [YYYY-MM-DD hh:mm:ss,iii] LEVEL: <message>
    reg = re.compile('^((?:/[^/]+)+)\\s+:\\s+\[([0-9\-: ,]+)]'
                     '\\s+(DEBUG|INFO|WARNING|ERROR):\\s+(.+)')

    match = reg.search(line)

    if match:
        curpath, timestamp, level, message = match.groups()
        path_parts = PurePath(curpath).relative_to('/groups/CaiLab/analyses').parts

        user, dataset, analysis, position = path_parts[:4]
        timestamp = dateutil.parser.isoparse(timestamp)

    else:
        return None

    return (
        user, dataset, analysis, position,
        datetime.strftime(timestamp, '%Y/%m/%d %H:%M:%S'), level, message
    )


def parse_log_file(fname, nlines=20):
    log = open(fname, 'r')
    lines = log.readlines()[-nlines:]

    parsed_lines = np.flip([parse_line(line) for line in lines], axis=0)

    result = pd.DataFrame.from_records(
        parsed_lines,
        columns=['User', 'Dataset', 'Analysis',
                 'Position', 'Time', 'level', 'Message']
    )

    return result


@app.callback(
    Output('mo-table-wrapper', 'children'),
    Input('mo-interval', 'n_intervals')
)
def update_logging_table(n_intervals):
    err = data_client.client.download_s3_objects(
        data_client.bucket_name,
        data_client.analysis_log
    )

    log_df = parse_log_file(data_client.analysis_log, nlines=400).drop(
        columns=['level']
    )

    return dbc.Table.from_dataframe(
        log_df,
        striped=True,
        size='sm',
        style={'font-size': '10pt'}
    )


layout = [
    dbc.Col([
        dcc.Interval(id='mo-interval', n_intervals=0, interval=30*1000),
        dbc.Card([
            dbc.CardHeader('Log monitoring'),
            dbc.CardBody(id='mo-table-wrapper', style={'height': '600px', 'overflow': 'scroll'})
        ])
    ])
]
