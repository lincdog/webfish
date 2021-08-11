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
from pages.common import ComponentManager, data_client, get_all_datasets
from pages._submission_util import SubmissionHelper


all_datasets = get_all_datasets()

pipeline_stages = [
    'alignment',
    'preprocessing',
    'dot detection',
    'segmentation',
    'decoding',
    'segmentation'
]
stage_ids = [
    'alignment',
    'preprocessing',
    'dot detection',
    'segmentation',
    'decoding',
    'segmentation-advanced'
]
stage_headers = [
    'Alignment Options',
    'Preprocessing Options',
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

    'sb-one-z-select':
        dbc.FormGroup([
            dbc.Checklist(
                id='sb-one-z-select',
                options=[{'label': 'Single Z slice', 'value': '1'}],
                value=[],
                switch=True
            ),
            dbc.FormText('Check this box if your images contain only '
                         'a single Z slice.')
        ]),

    # stage selection
    'sb-stage-select':
        dbc.FormGroup([
            dbc.Label('Select pipeline stages to run', html_for='sb-stage-select'),
            dbc.Checklist(
                id='sb-stage-select',
                options=[
                    {'label': 'Alignment', 'value': 'alignment'},
                    {'label': 'Preprocessing', 'value': 'preprocessing'},
                    {'label': 'Dot Detection', 'value': 'dot detection'},
                    {'label': 'Segmentation', 'value': 'segmentation'},
                    {'label': 'Decoding', 'value': 'decoding'}
                ],
                value=['alignment', 'preprocessing', 'dot detection', 'segmentation', 'decoding'],
                switch=True
            ),
            dbc.FormText('Select which stage(s) of the pipeline you want to run. ')
        ]),

    'sb-preprocessing-checklist':
        dbc.Checklist(
            id='sb-preprocessing-checklist',
            options=[
                {'label': 'Tophat transform before BG subtract',
                 'value': 'tophat raw data kernel size'},
                {'label': 'Dilate background image before subtraction',
                 'value': 'dilate background kernel'},
                {'label': 'Run background subtraction (with final_background image)',
                 'value': 'background subtraction'},
                {'label': 'Tophat transform after BG subtract (removes large, bright objects)',
                 'value': 'tophat preprocessing'},
                {'label': 'Rolling ball background subtraction',
                 'value': 'rolling ball preprocessing'},
                {'label': 'Apply slight blur (remove hot pixels)',
                 'value': 'blur preprocessing'},
            ],
            value=['background subtraction',
                   'tophat preprocessing',
                   'rolling ball preprocessing',
                   'blur preprocessing'],
            switch=True
        ),

    'sb-tophat-kernel-size':
        dbc.FormGroup([
            dbc.Label('Tophat kernel size', html_for='sb-tophat-kernel-size'),
            dbc.Input(
                id='sb-tophat-kernel-size',
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
                max=2000,
                step=1,
                value=1000,
                disabled=False
            ),
            dbc.FormText('Sets the rolling ball kernel radius. A smaller value '
                         'generally results in higher background estimated at each pixel'
                         ' and more variance of the estimated background.')
        ]),
    'sb-blur-kernel-size':
        dbc.FormGroup([
            dbc.Label('Blur kernel size', html_for='sb-blur-kernel-size'),
            dbc.Input(
                id='sb-blur-kernel-size',
                type='number',
                min=0,
                max=20,
                step=1,
                value=3,
                disabled=False
            ),
            dbc.FormText('Sets the blur kernel radius in pixels. In general, '
                         'features smaller than this radius will be removed or '
                         'attenuated by the blur. So it should be just below the '
                         'expected real dot size.')
        ]),

    # alignment
    'sb-alignment-select':
        dbc.FormGroup([
            dbc.Label('Alignment Algorithm', html_for='sb-alignment-select'),
            dbc.Select(
                id='sb-alignment-select',
                value='matlab dapi',
                options=[{'label': 'DAPI Alignment', 'value': 'matlab dapi'}],
                disabled=False
            ),
            dbc.FormText(
                dcc.Markdown(
                    'The underlying MATLAB function is '
                    '[imregtform](https://www.mathworks.com/help/images/ref/imregtform.html)'),
                style={'font-size': '10pt'}
            )
        ]),

    # dot detection
    'sb-dot detection-select':
        dbc.FormGroup([
            dbc.Label('Dot Detection Algorithm', html_for='sb-dot detection-select'),
            dbc.Select(
                id='sb-dot detection-select',
                options=[
                    {'label': 'Biggest Jump Python', 'value': 'biggest jump 3d'},
                    {'label': 'ADCG 2D', 'value': 'adcg 2d'},
                    {'label': 'Biggest Jump Matlab', 'value': 'matlab 3d'}
                ],
                value='biggest jump 3d',
                disabled=False
            ),
        ]),

    'sb-strictness-select':
        dbc.FormGroup([
            dbc.Label('Strictness parameter', html_for='sb-strictness-select'),
            dcc.Slider(
                id='sb-strictness-select',
                min=-10,
                max=25,
                step=1,
                value=2,
                marks={i: str(i) for i in range(-10, 26, 3)}
            ),
            dbc.FormText('Higher strictness sets a '
                         'higher minimum intensity threshold for dot detection.')
        ]),

    'sb-remove-bright-dots':
        dbc.FormGroup([
            dbc.Checklist(
                id='sb-remove-bright-dots',
                options=[
                   {'label': 'Keep abnormally bright dots',
                    'value': 'keep'}
                ],
                value=['keep'],
                switch=True
                ),
            dbc.FormText('Select to preserve dots which are much brighter than '
                         'the other dots in the image. Deselect to throw these '
                         'out.')
        ]),

    'sb-threshold-select':
        dbc.FormGroup([
            dbc.Label('Laplacian Threshold', html_for='sb-threshold-select'),
            dbc.Input(
                id='sb-threshold-select',
                type='number',
                min=0,
                max=0.05,
                step=0.0001,
                value=0.0001,
                disabled=False
            ),
            dbc.FormText('Set a threshold of the LoG filter. Default is usually fine.')
        ]),
    'sb-minsigma-dotdetection':
        dbc.FormGroup([
            dbc.Label('Minimum sigma for Python dot detection',
                      html_for='sb-minsigma-dotdetection'),
            dbc.Input(
                id='sb-minsigma-dotdetection',
                type='number',
                min=0,
                max=20,
                step=0.1,
                value=1
            ),
            dbc.FormText(dcc.Markdown(
                'Sets the `min_sigma` parameter for [`skimage.feature.blob_log`]'
                '(https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log), '
                'in pixels. Defines the rough minimum radius of detected dots'))
        ]),
    'sb-maxsigma-dotdetection':
        dbc.FormGroup([
            dbc.Label('Maximum sigma for Python dot detection',
                      html_for='sb-maxsigma-dotdetection'),
            dbc.Input(
                id='sb-maxsigma-dotdetection',
                type='number',
                min=0,
                max=20,
                step=0.1,
                value=2
            ),
            dbc.FormText(dcc.Markdown(
                'Sets the `max_sigma` parameter for [`skimage.feature.blob_log`]'
                '(https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log), '
                'in pixels. Defines the rough maximum radius of detected dots'))
        ]),
    'sb-numsigma-dotdetection':
        dbc.FormGroup([
            dbc.Label('Number of sigma steps for Python dot detection',
                      html_for='sb-numsigma-dotdetection'),
            dbc.Input(
                id='sb-numsigma-dotdetection',
                type='number',
                min=0,
                max=20,
                step=1,
                value=2
            ),
            dbc.FormText(dcc.Markdown(
                'Sets the `num_sigma` parameter for [`skimage.feature.blob_log`]'
                '(https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log), '
                'the number of sigma steps between `min_sigma` and `max_sigma` that the '
                'function uses for detecting dots.'))
        ]),
    'sb-overlap-dotdetection':
        dbc.FormGroup([
            dbc.Label('Overlap parameter for Python dot detection',
                      html_for='sb-overlap-dotdetection'),
            dbc.Input(
                id='sb-overlap-dotdetection',
                type='number',
                min=0,
                max=1,
                step=0.01,
                value=0.5
            ),
            dbc.FormText(dcc.Markdown(
                'Sets the `overlap` parameter for [`skimage.feature.blob_log`]'
                '(https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log), '
                ' from 0-1. Defines maximum allowed overlap between two candidate dots.'))
        ]),

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

    # Decoding
    'sb-decoding-algorithm':
        dbc.FormGroup([
            dbc.Label('Select Decoding Algorithm', html_for='sb-decoding-algorithm'),
            dbc.RadioItems(
                id='sb-decoding-algorithm',
                options=[
                    {'label': 'Standard MATLAB', 'value': 'matlab'},
                    {'label': 'Syndrome decoding', 'value': 'syndrome'}
                ],
                value='matlab',
            ),
            dbc.FormText('Select which decoding algorithm to use. The barcode key '
                         'will be the same for both, but see the Home page instructions '
                         'for how to specify the parity check table for Syndrome.')
        ]),
    'sb-decoding-select':
        dbc.FormGroup([
            dbc.Label('Select Decoding Scheme', html_for='sb-decoding-select'),
            dbc.RadioItems(
                id='sb-decoding-select',
                options=[
                    {'label': 'Across channels', 'value': 'across'},
                    {'label': 'Individual channel(s)', 'value': 'individual'},
                    {'label': 'smFISH (non-barcoded)', 'value': 'non barcoded'}
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
                    {'label': str(i), 'value': str(i+1)}
                    for i in range(5)
                ],
                id='sb-individual-channel-select',
                inline=True
            ),
            dbc.FormText('This only has an effect if "Individual" is selected above.')
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
excluded_status_comps = ['sb-stage-select', 'sb-segmentation-label']

component_groups = {
    'basic-metadata': ['sb-analysis-name',
                       'sb-position-select',
                       'sb-one-z-select',
                       'sb-stage-select'],

    'alignment': ['sb-alignment-select'],

    'preprocessing': ['sb-preprocessing-checklist',
                      'sb-tophat-kernel-size',
                      'sb-rollingball-kernel-size',
                      'sb-blur-kernel-size'],

    'dot detection': ['sb-dot detection-select',
                      'sb-strictness-select'],

    'dot detection-python': ['sb-remove-bright-dots',
                             'sb-threshold-select',
                             'sb-minsigma-dotdetection',
                             'sb-maxsigma-dotdetection',
                             'sb-numsigma-dotdetection',
                             'sb-overlap-dotdetection'],

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

    'decoding': ['sb-decoding-algorithm',
                 'sb-decoding-select',
                 'sb-individual-channel-select'],

    'decoding-syndrome': ['sb-syndrome-lateral-variance',
                          'sb-syndrome-z-variance',
                          'sb-syndrome-logweight-variance']

}


cm = ComponentManager(clear_components, component_groups=component_groups)
helper = SubmissionHelper(data_client, cm)  # No default graph or logger needed


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


# This is the callback that gathers all the form statuses and begins the process
# of converting them to the JSON file. Its main function is to validate the input,
# show any errors if present, and raise the modal confirm dialog if no errors.
@app.callback(
    Output('sb-modal-container', 'children'),
    Input('sb-submit-button', 'n_clicks'),
    State('user-select', 'value'),
    State('dataset-select', 'value'),
    State('sb-stage-select', 'value'),
    # Get the state of every form component except for those in excluded_status_comps
    # Note these are gathered in an *arg called values.
    [State(comp, 'value') for comp in cm.clear_components.keys()
     if comp not in excluded_status_comps]
)
def submit_new_analysis(n_clicks, user, dataset, stage_select, *values):
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
    # and its validation functions may depend on the state of the
    status.update({c: v for c, v in zip(relevant_comps, values)})

    # This performs the validation and conversion of the form IDs to the appropriate
    # JSON keys for the pipeline.
    analysis_name, submission = helper.form_to_json_output(status, stage_select)
    # Any errors are added under this key.
    sub_errors = submission.pop('__ERRORS__')

    # Remove all form values from pipeline stages that are deselected.
    for stage in set(pipeline_stages):
        if stage not in stage_select:
            submission.pop(stage, None)

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
                html.Summary('Generated JSON'),
                # This Pre component serves as the intermediate JSON contents before
                # final submission. Its contents are used by upload_generated_json
                # to upload the JSON to S3.
                html.Pre(json.dumps(submission, indent=2), id='sb-generated-json')
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
        submission_bytes = io.BytesIO(json.dumps(new_dict).encode())

        data_client.client.client.upload_fileobj(
            submission_bytes,
            Bucket=data_client.bucket_name,
            Key=f2k(Path('json_analyses', analysis_name + '.json'))
        )
        alerts.append(
            dbc.Alert(
                dcc.Markdown(f'Successfully uploaded new analysis `{analysis_name}`!'
                             f' The full path will be: '
                             f'`{user}/{dataset}/{analysis_name}`'),
                color='success'))

    except Exception as e:
        alerts.append(dbc.Alert('Error uploading JSON to S3: ', e, color='danger'))

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
        [html.Summary('Alignment options')] +
        cm.component_group('alignment', tolist=True),
        open=True, id='sb-alignment-wrapper'
    ),
    html.Hr(),
    html.Details(
        [html.Summary('Preprocessing options')] +
        cm.component_group('preprocessing', tolist=True),
        open=True, id='sb-preprocessing-wrapper'
    )
]

col2_clear = [
    html.Details(
        [html.Summary('Dot detection options')] +
        cm.component_group('dot detection', tolist=True),
        open=True, id='sb-dot detection-wrapper'
    ),
    html.Details(
        [html.Summary('Dot detection options (Python dot detection only)')] +
        cm.component_group('dot detection-python', tolist=True)
    )
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
        [html.Summary('Decoding options')] +
        cm.component_group('decoding', tolist=True),
        open=True, id='sb-decoding-wrapper'
    ),
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


