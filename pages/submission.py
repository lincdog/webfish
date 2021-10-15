import numpy as np
import pandas as pd
import io
import json
import yaml
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
from pages.common import ComponentManager, data_client, get_all_datasets
from pages._submission_util import SubmissionHelper, _add_c


all_datasets = get_all_datasets()

channels = [730, 640, 561, 488, 405]

pipeline_stages = [
    'preprocessing',
    'segmentation',
    'segmentation'
]
stage_ids = [
    'preprocessing',
    'segmentation',
    'segmentation-advanced'
]
stage_headers = [
    'Preprocessing Options',
    'Segmentation Options',
    'Advanced Segmentation Options',
]


def make_channel_specific_form(channel, tolist=False):
    def add_c(s):
        return _add_c(s, channel)

    output_dict = {
        'sb-sigma-upper-bound':
            dbc.FormGroup([
                dbc.Label('Sigma upper bound (pixels)',
                          html_for=add_c('sb-sigma-upper-bound')),
                dbc.Input(
                    id=add_c('sb-sigma-upper-bound'),
                    type='number',
                    min=0.5,
                    max=15.0,
                    step=0.1,
                    value=2.3,
                    disabled=False
                ),
                dbc.FormText('Sets the maximum sigma value for ADCG to use.')
            ]),

        'sb-sigma-lower-bound':
            dbc.FormGroup([
                dbc.Label('Sigma lower bound (pixels)',
                          html_for=add_c('sb-sigma-lower-bound')),
                dbc.Input(
                    id=add_c('sb-sigma-lower-bound'),
                    type='number',
                    min=0,
                    max=10.0,
                    step=0.1,
                    value=0.8,
                    disabled=False
                ),
                dbc.FormText('Sets the minimum sigma value for ADCG to use.')
            ]),

        'sb-min-weight':
            dbc.FormGroup([
                dbc.Label('Minimum weight (threshold)',
                          html_for=add_c('sb-min-weight')),
                dbc.Input(
                    id=add_c('sb-min-weight'),
                    type='number',
                    min=0,
                    max=20000,
                    step=10,
                    value=120,
                    disabled=False
                ),
                dbc.FormText('Sets the absolute minimum dot intensity for ADCG to '
                             'consider.')
            ]),

        add_c('sb-final-loss-improvement'):
            dbc.FormGroup([
                dbc.Label('ADCG Final loss improvement',
                          html_for=add_c('sb-final-loss-improvement')),
                dbc.Input(
                    id=add_c('sb-final-loss-improvement'),
                    type='number',
                    min=1.,
                    max=2000.,
                    step=1,
                    value=70.0,
                    disabled=False
                ),
                dbc.FormText('Sets the loss improvement at which ADCG will terminate.')
            ]),
        # End channel-specific parameters
    }

    if tolist:
        return list(output_dict.values())
    else:
        return output_dict


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
                placeholder='Select one or more positions'
            ),
            dbc.FormText('Select one or more positions')
        ]),

    'sb-channel-select':
        dbc.FormGroup([
            dbc.Label('Select Channel(s) for dot detection and decoding',
                      html_for='sb-channel-select'),
            dcc.Dropdown(
                options=[
                    {'label': str(i), 'value': i}
                    for i in channels
                ],
                id='sb-channel-select',
                multi=True,
                placeholder='Select one or more channels'
            ),
        ]),

    'sb-alignment-channel-select':
        dbc.FormGroup([
            dbc.Label('Select Channel to use for alignment',
                      html_for='sb-alignment-channel-select'),
            dcc.Dropdown(
                options=[
                    {'label': str(i), 'value': i}
                    for i in channels
                ],
                value=405,
                id='sb-alignment-channel-select',
                placeholder='Select one channel',
                clearable=False
            ),
        ]),

    'sb-z-slice-select':
        dbc.FormGroup([
            dbc.Label('Z slice to process', html_for='sb-z-slice-select'),
            dbc.Input(
                id='sb-z-slice-select',
                type='number',
                min=0,
                max=30,
                step=1,
                value=0,
                disabled=False
            ),
            dbc.FormText('Select which Z slice of the images to process. '
                         'Be careful not to choose one beyond the Z length of '
                         'your images.')
        ]),

    # Preprocessing
    'sb-median-kernel-size':
        dbc.FormGroup([
            dbc.Label('Median filter kernel size', html_for='sb-median-kernel-size'),
            dbc.Input(
                id='sb-median-kernel-size',
                type='number',
                min=0,
                max=200,
                step=1,
                value=10,
                disabled=False
            ),
            dbc.FormText('Sets the tophat kernel size in pixels. This should be '
                         'between the expected dot radius and the expected radius of '
                         'undesirable large blobs such as autofluorescence.')
        ]),
    'sb-rollingball-kernel-size':
        dbc.FormGroup([
            dbc.Label('Rolling ball kernel size', html_for='sb-rollingball-kernel-size'),
            dbc.Input(
                id='sb-rollingball-kernel-size',
                type='number',
                min=1,
                max=30,
                step=0.1,
                value=3.3,
                disabled=False
            ),
            dbc.FormText('Sets the rolling ball kernel radius. A smaller value '
                         'generally results in higher background estimated at each pixel'
                         ' and more variance of the estimated background.')
        ]),

    # ADCG dot detection
    'sb-max-iters':
        dbc.FormGroup([
            dbc.Label('Max iterations', html_for='sb-max-iters'),
            dbc.Input(
                id='sb-max-iters',
                type='number',
                min=0,
                max=2000,
                step=1,
                value=100,
                disabled=False
            ),
            dbc.FormText('Sets the maximum number of ADCG optimization iterations.'
                         )
        ]),
    'sb-max-cd-iters':
        dbc.FormGroup([
            dbc.Label('Max CD iterations', html_for='sb-max-cd-iters'),
            dbc.Input(
                id='sb-max-cd-iters',
                type='number',
                min=0,
                max=100,
                step=1,
                value=10,
                disabled=False
            ),
            dbc.FormText('Sets the maximum number of ADCG '
                         'conditional descent iterations.'
                         )
        ]),
    'sb-min-allowed-separation':
        dbc.FormGroup([
            dbc.Label('Minimum allowed separation',
                      html_for='sb-min-allowed-separation'),
            dbc.Input(
                id='sb-min-allowed-separation',
                type='number',
                min=0.1,
                max=20,
                step=0.1,
                value=2.0,
                disabled=False
            ),
            dbc.FormText('Sets the minimum distance two dots can be apart without '
                         'being removed.')
        ]),

    # Channel-specific parameters here

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
            dbc.FormText(dcc.Markdown('For more information about Cellpose: '
                         'https://github.com/MouseLand/cellpose'))
        ]),
    'sb-segmentation-checklist':
        dbc.Checklist(
            id='sb-segmentation-checklist',
            options=[
                {'label': 'Only decode dots in cells',
                 'value': 'only decode dots in cells'},
                {'label': 'Run all post-analyses',
                 'value': 'all post analyses'},
                {'label': 'Nuclear segmentation '
                          '(uses Labeled_Images directory if present)',
                 'value': 'nuclei labeled image'},
                {'label': 'Cytoplasm segmentation '
                          '(uses Labeled_Images_Cytoplasm directory if present)',
                 'value': 'cyto labeled image'},
                {'label': 'Match nuclear and cytoplasm segmentation',
                 'value': 'nuclei cyto match'}
            ],
            value=['only decode dots in cells',
                   'all post analyses',
                   ],
            switch=True
        ),
    'sb-segmentation-label':
        dbc.Label(
            '"Match nuclear and cytoplasm segmentation" traces each nucleus '
            'mask back to a cytoplasm mask in the same [x, y] '
            'location and only keeps the nuclei and cytoplasm that have a match',
            id='sb-segmentation-label', size='sm', style={'margin-left': '20px'}
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
                options=[{'label': str(i), 'value': str(i)}
                         for i in range(6)
                         ],
                value='3',
                inline=True
            ),
            dbc.FormText('Select which channel to use for cytoplasm segmentation')
        ]),
    'sb-nuclei-channel':
        dbc.FormGroup([
            dbc.Label('Nucleus Channel', html_for='sb-nuclei-channel'),
            dbc.RadioItems(
                id='sb-nuclei-channel',
                options=[{'label': str(i), 'value': str(i)}
                         for i in range(6)
                         ],
                value='3',
                inline=True
            ),
            dbc.FormText('Select which channel to use for nuclei segmentation')
        ]),
    'sb-nuclei-radius':
        dbc.FormGroup([
            dbc.Label('Select Nuclear Radius', html_for='sb-nuclei-radius'),
            dbc.Input(
                id='sb-nuclei-radius',
                type='number',
                min=0,
                max=1000,
                step=1,
                value=0
            ),
            dbc.FormText('Set to 0 to let Cellpose find automatically.')
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
    'sb-cyto-radius':
        dbc.FormGroup([
            dbc.Label('Select Cytoplasm Radius', html_for='sb-cyto-radius'),
            dbc.Input(
                id='sb-cyto-radius',
                type='number',
                min=0,
                max=1000,
                step=1,
                value=0
            ),
            dbc.FormText('Set to 0 to let Cellpose find automatically.')
        ]),
    'sb-cyto-cell-prob-threshold':
        dbc.FormGroup([
            dbc.Label('Set Cytoplasm Cell Probability Threshold',
                      html_for='sb-cyto-cell-prob-threshold'),
            dcc.Slider(
                id='sb-cyto-cell-prob-threshold',
                min=-6,
                max=6,
                step=1,
                value=-4,
                marks={i: str(i) for i in range(-6, 7, 1)}
            ),
        ]),
    'sb-cyto-flow-threshold':
        dbc.FormGroup([
            dbc.Label('Set Cytoplasm Flow Threshold Parameter',
                      html_for='sb-cyto-flow-threshold'),
            dcc.Slider(
                id='sb-cyto-flow-threshold',
                min=0,
                max=1,
                step=0.01,
                value=0.8,
                marks={i: f'{i:0.2f}' for i in np.linspace(0, 1, 11)}
            )
        ]),

    # decoding-syndrome
    'sb-syndrome-lateral-variance':
        dbc.FormGroup([
            dbc.Label('Lateral variance parameter',
                      html_for='sb-syndrome-lateral-variance'),
            dbc.Input(
                id='sb-syndrome-lateral-variance',
                type='number',
                min=0,
                value=200,
                step=0.01,
            )
        ]),
    'sb-syndrome-z-variance':
        dbc.FormGroup([
            dbc.Label('Z variance parameter',
                      html_for='sb-syndrome-z-variance'),
            dbc.Input(
                id='sb-syndrome-z-variance',
                type='number',
                min=0,
                value=40,
                step=0.01,
            )
        ]),
    'sb-syndrome-logweight-variance':
        dbc.FormGroup([
            dbc.Label('Log weight variance parameter',
                      html_for='sb-syndrome-logweight-variance'),
            dbc.Input(
                id='sb-syndrome-logweight-variance',
                type='number',
                min=0,
                value=0,
                step=0.01,
            )
        ])


}


# The component id's that are NOT used in forming the JSON output upon submission.
# The submit callback checks the value of every component EXCEPT these.
excluded_status_comps = ['sb-segmentation-label']

channel_specific_comps = ['sb-sigma-upper-bound',
                          'sb-sigma-lower-bound',
                          'sb-min-weight',
                          'sb-final-loss-improvement']

component_groups = {
    'basic-metadata': ['sb-analysis-name',
                       'sb-position-select',
                       'sb-channel-select',
                       'sb-alignment-channel-select',
                       'sb-z-slice-select'],

    'preprocessing': ['sb-median-kernel-size',
                      'sb-rollingball-kernel-size'],

    'adcg-channel-specific': channel_specific_comps,

    'adcg-general': ['sb-max-iters',
                     'sb-max-cd-iters',
                     'sb-min-allowed-separation'],

    'segmentation': ['sb-segmentation-select',
                     'sb-segmentation-checklist',
                     'sb-segmentation-label'],

    'segmentation-advanced': ['sb-edge-deletion',
                              'sb-nuclei-distance',
                              'sb-cyto-channel',
                              'sb-cyto-radius',
                              'sb-cyto-cell-prob-threshold',
                              'sb-cyto-flow-threshold',
                              'sb-nuclei-channel',
                              'sb-nuclei-radius',
                              'sb-cell-prob-threshold',
                              'sb-flow-threshold'],

    'decoding-syndrome': ['sb-syndrome-lateral-variance',
                          'sb-syndrome-z-variance',
                          'sb-syndrome-logweight-variance']

}


cm = ComponentManager(clear_components, component_groups=component_groups)
helper = SubmissionHelper(data_client, cm)  # No default graph or logger needed


@app.callback(
    Output('sb-position-select', 'options'),
    Output('sb-position-select', 'value'),
    Input('user-select', 'value'),
    Input('dataset-select', 'value')
)
def select_user_dataset(user, dataset):
    if not user or not dataset:
        raise PreventUpdate

    query = 'user==@user and dataset==@dataset and source_key=="hyb_fov"'

    positions = np.sort(data_client.datafiles.query(
        query)['position'].unique().astype(int))

    return [{'label': p, 'value': p} for p in list(positions)], list(positions)

    #analyses = data_client.datafiles.query(
    #    'user==@user and dataset==@dataset')['analysis'].unique()


@app.callback(
    Output('sb-channel-options-wrapper', 'children'),
    Input('sb-channel-select', 'value'),
    State('sb-channel-options-wrapper', 'children')
)
def select_channels(selected_channels, current_contents):

    print(f'{selected_channels=}')

    if not selected_channels:
        raise PreventUpdate

    selected_channels = sorted(selected_channels)
    selected_channels.reverse()

    cards = []

    for chan in selected_channels:
        # Add the form components for each channel-specific parameter
        form_components = make_channel_specific_form(chan, tolist=True)

        channel_entry = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(f'Options for {chan}'),
                    *form_components
                ]
            ),
            style={"width": "40%"})

        cards.append(channel_entry)

    return dbc.CardGroup(cards)


# This is the callback that gathers all the form statuses and begins the process
# of converting them to the JSON file. Its main function is to validate the input,
# show any errors if present, and raise the modal confirm dialog if no errors.
@app.callback(
    Output('sb-modal-container', 'children'),
    Input('sb-submit-button', 'n_clicks'),
    State('user-select', 'value'),
    State('dataset-select', 'value'),
    # Get the state of every form component except for those in excluded_status_comps
    # Note these are gathered in an *arg called values.
    [State(comp, 'value') for comp in cm.clear_components.keys()
     if comp not in excluded_status_comps]
)
def submit_new_analysis(n_clicks, user, dataset, *values):
    if not n_clicks:
        raise PreventUpdate

    upload = True

    # The form components that matter for the JSON file
    relevant_comps = [c for c in cm.clear_components.keys()
                      if c not in excluded_status_comps]

    # We always start with the user and dataset, which are not part of
    # relevant_comps because they are from index.py.
    status = {'user-select': user, 'dataset-select': dataset}

    # Add the status of every relevant form component to the status dict
    # Note the order is determined by the components' order in cm.clear_components.
    # This matters because helper.form_to_json_output loops through them in this order
    # and its validation functions may depend on the state of previous entries
    status.update({c: v for c, v in zip(relevant_comps, values)})

    selected_channels = status['sb-channel-select']
    #FIXME: Not able to get the right inputs because they would need to be supplied
    #  as arguments to the decorator, before the function is entered
    #for chan in selected_channels:
    #    channel_keys = [_add_c(k, chan) for k in channel_specific_comps]
    #    status.update({c: })


    # This performs the validation and conversion of the form IDs to the appropriate
    # JSON keys for the pipeline.
    analysis_name, submission = helper.form_to_json_output(status)
    # Any errors are added under this key.
    sub_errors = submission.pop('__ERRORS__')

    analyses = all_datasets.query(
        'user==@user and dataset==@dataset')['analysis'].unique()

    alerts = []

    # If there are any validation errors, add them as dbc.Alerts, set the upload flag False
    if len(sub_errors) > 0:
        upload = False
        alerts.extend([
            dbc.Alert(f'Validation error: {e}', color='danger')
            for e in sub_errors
        ])

    # Make sure we have user, dataset and analysis name.
    if not all((analysis_name, user, dataset)):
        upload = False
        alerts.append(dbc.Alert('Please choose a user and dataset, and specify'
                                ' a new analysis name.', color='danger'))

    # Make sure the analysis name is unique as far as we know.
    if analysis_name in analyses:
        upload = False
        alerts.append(
            dbc.Alert(dcc.Markdown(f'An analysis named `{analysis_name}` '
                      f'already exists for the chosen dataset.'), color='danger'))

    if not upload:
        alerts.append(dbc.Alert('Did not upload JSON to S3 due to errors', color='danger'))

    # We want the most general errors first, but we have to check the specific
    # ones first.
    alerts.reverse()

    modal = dbc.Modal([
        dbc.ModalHeader('Confirm Submission'),
        dbc.ModalBody([
            'Press "Confirm and Submit" to send the analysis request, '
            'or "Go back" to go back and make changes.',
            html.H4(html.Code(analysis_name, id='sb-final-analysis-name')),
            html.Details([
                html.Summary('Generated YAML'),
                # This Pre component serves as the intermediate JSON contents before
                # final submission. Its contents are used by upload_generated_json
                # to upload the JSON to S3.
                html.Pre(yaml.dump(submission), id='sb-generated-json')
                ], id='sb-submission-json', open=True),
            dbc.Button('Confirm and Submit',
                       id='sb-confirm-submit-button',
                       color='success',
                       n_clicks=0,
                       style={'margin': '5px'}),
            dbc.Button('Go back',
                       id='sb-go-back-button',
                       color='secondary',
                       n_clicks=0,
                       style={'margin': '5px'})
        ])
    ], id='sb-confirm-modal',
        backdrop='static',
        centered=True,
        is_open=True,
        scrollable=True
    )

    # If we are all good, return the confirm modal. If there were errors,
    # return them instead.
    if upload:
        return modal
    else:
        return alerts


@app.callback(
    Output('sb-submission-alerts', 'children'),
    Output('sb-confirm-modal', 'is_open'),
    Input('sb-confirm-submit-button', 'n_clicks'),
    Input('sb-go-back-button', 'n_clicks'),
    State('sb-confirm-modal', 'is_open'),
    State('sb-generated-json', 'children'),
    State('user-select', 'value'),
    State('dataset-select', 'value'),
    State('sb-final-analysis-name', 'children'),
)
def upload_generated_json(
    n_submit_clicks,
    n_go_back_clicks,
    modal_state,
    new_json,
    user,
    dataset,
    analysis_name
):
    error = None
    alerts = []

    if not modal_state and (n_submit_clicks or n_go_back_clicks):
        raise PreventUpdate

    if n_go_back_clicks > 0:
        return [], False

    if n_submit_clicks == 0:
        raise PreventUpdate

    try:
        new_dict = json.loads(new_json, parse_int=str, parse_float=str)
    except Exception as e:
        error = e
        alerts.append(dbc.Alert(f'There was a problem reading the generated JSON '
                                f'from the DOM: {error}', color='danger'))

        return alerts, False

    try:
        submission_bytes = io.BytesIO(yaml.dump(new_dict).encode())

        data_client.client.client.upload_fileobj(
            submission_bytes,
            Bucket=data_client.bucket_name,
            Key=f2k(Path('yaml_configs', f'{analysis_name}_config.yaml'))
        )
        alerts.append(
            dbc.Alert(
                dcc.Markdown(f'Successfully uploaded new analysis `{analysis_name}`!'
                             f' The full path will be: '
                             f'`{user}/{dataset}/{analysis_name}`'),
                color='success'))

    except Exception as e:
        alerts.append(dbc.Alert('Error uploading yaml to S3: ', e, color='danger'))

    return alerts, False


@app.callback(
    [Output('sb-col-1', 'children'),
     Output('sb-col-2', 'children')],
    Input('sb-reset-button', 'n_clicks')
)
def reset_to_defaults(n_clicks):
    if n_clicks:
        return col1_clear, col2_clear, col3_clear
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
        [html.Summary('Preprocessing options')] +
        cm.component_group('preprocessing', tolist=True),
        open=True, id='sb-preprocessing-wrapper'
    )
]

col2_clear = [
    html.Div([
            dcc.Loading(id='sb-channel-options-wrapper')
        ],
        id='sb-channel-options-div'
    ),
    html.Hr(),
    html.Div(
        cm.component_group('adcg-general', tolist=True),
        id='sb-adcg-options-wrapper'
    ),
]

col3_clear = [
    html.Details(
        [html.Summary('Segmentation Options')] +
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
        [html.Summary('Decoding options (Syndrome decoding only)')] +
        cm.component_group('decoding-syndrome', tolist=True)
    ),
    dbc.Card([
        dbc.CardHeader('Submission'),
        dbc.CardBody([
            dbc.Button('Submit', id='sb-submit-button', color='primary', size='lg'),
            dbc.Button('Reset to defaults', id='sb-reset-button', color='warning', size='lg'),
            html.Div(id='sb-submission-alerts', style={'margin': '10px'}),
            html.Div(id='sb-modal-container', style={'margin': '10px'}),
        ]),
    ]),
]

layout = [
    dbc.Col(col1_clear, id='sb-col-1', width=4),
    dbc.Col(col2_clear, id='sb-col-2', width=4),
    dbc.Col(col3_clear, id='sb-col-3', width=4)
]


