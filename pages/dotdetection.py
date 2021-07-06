import dash
import numpy as np

import logging

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import no_update

from app import app
from lib.util import (
    safe_imread,
    pil_imread,
    sort_as_num_or_str,
)

from pages._page_util import (
    aggregate_dot_dfs,
    DotDetectionHelper,
)
from pages.common import ComponentManager, data_client

logger = logging.getLogger('webfish.' + __name__)


clear_components = {

    'dd-analysis-select':
        dbc.Select(
            id='dd-analysis-select',
            placeholder='Select an analysis',
            persistence=True,
            persistence_type='session'
        ),

    'dd-hyb-select-label': dbc.Label('Select a hyb round', html_for='dd-hyb-select'),
    'dd-hyb-select':
        dbc.Select(id='dd-hyb-select', placeholder='Select a hyb round'),
    'dd-position-select-label': dbc.Label('Select a position', html_for='dd-position-select'),
    'dd-position-select':
        dbc.Select(id='dd-position-select', placeholder='Select a position'),

    'dd-z-cap': html.B('Select Z slice'),
    'dd-chan-cap': 'Select a channel',
    'dd-contrast-cap': html.B('Adjust contrast'),
    'dd-z-select': dcc.Slider(id='dd-z-select'),
    'dd-chan-select': dbc.Select(
        id='dd-chan-select',
        options=[{'label': str(c), 'value': str(c)} for c in range(5)]
        ),
    'dd-contrast-slider': dcc.RangeSlider(id='dd-contrast-slider'),
    'dd-contrast-note': dcc.Markdown('NOTE: the image intensity is rescaled to '
                                     'use the full range of the datatype before '
                                     'display'),
    'dd-strictness-slider': dbc.FormGroup([
        dbc.Label('Strictness filter', html_for='dd-strictness-slider'),
        dcc.Slider(
            id='dd-strictness-slider',
            min=-20,
            max=100,
            step=1,
            value=0,
            marks={i: str(i) for i in range(-20, 101, 10)}
        )
    ]),

    'dd-fig': dcc.Graph(
        id='dd-fig',
        config={
            'scrollZoom': True,
            'modeBarButtonsToRemove': ['zoom2d', 'zoomOut2d', 'zoomIn2d']
        }
    )
}

component_groups = {
    'dataset-info': ['dd-analysis-select'],

    'image-select': ['dd-hyb-select-label',
                     'dd-hyb-select',
                     'dd-position-select-label',
                     'dd-position-select',
                     'dd-chan-cap',
                     'dd-chan-select',
                     ],

    'image-params': ['dd-z-cap',
                     'dd-z-select',
                     'dd-contrast-cap',
                     'dd-contrast-slider',
                     'dd-contrast-note']
}

cm = ComponentManager(clear_components, component_groups=component_groups)
helper = DotDetectionHelper(data_client, cm, 'dd-fig', logger)


@app.callback(
    Output('dd-graph-wrapper', 'children'),
    Input('dd-detail-tabs-collapse', 'is_open'),
    Input('dd-detail-tabs', 'active_tab'),
    Input('dd-z-select', 'value'),
    Input('dd-chan-select', 'value'),
    Input('dd-contrast-slider', 'value'),
    Input('dd-strictness-slider', 'value'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    Input('dd-analysis-select', 'value'),
    Input('dataset-select', 'value'),
    Input('user-select', 'value'),
    State('dd-fig', 'relayoutData')
)
def update_visualization(
    is_open,
    active_tab,
    z,
    channel,
    contrast,
    strictness,
    position,
    hyb,
    analysis,
    dataset,
    user,
    current_layout
):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.info(f'Entering update_image_params with trigger {trigger}')

    # Again test for None explicitly because z, channel, position or hyb might be 0
    if not is_open or any([v is None for v in
                           (active_tab, position, hyb, dataset, user)]):
        return helper.dash_graph()

    if active_tab == 'dd-tab-dotdetection':
        return helper.prepare_dotdetection_figure(
            z, channel, contrast, strictness, position,
            hyb, analysis, dataset, user, current_layout
        )
    elif active_tab == 'dd-tab-preprocess':
        return helper.prepare_preprocess_figure(
            position, hyb, channel,
            analysis, dataset, user
        )
    elif active_tab == 'dd-tab-alignment':
        return helper.prepare_alignment_figure(
            position, analysis, dataset, user
        )
    elif active_tab == 'dd-tab-locations':
        return helper.prepare_locations_figure(
            position, analysis, dataset, user
        )
    else:
        return helper.dash_graph()


@app.callback(
    Output('dd-dot-breakdown-wrapper', 'children'),
    Input('dd-detail-tabs-collapse', 'is_open'),
    Input('dd-hyb-select', 'value'),
    Input('dd-position-select', 'value'),
    Input('dd-analysis-select', 'value'),
    Input('dataset-select', 'value'),
    Input('user-select', 'value'),
)
def populate_breakdown_table(
    is_open,
    hyb,
    position,
    analysis,
    dataset,
    user
):

    if not all([hyb is not None, analysis, dataset, user]):
        return []

    request = {'user': user, 'dataset': dataset, 'analysis': analysis}

    if position is not None:
        request.update({'position': position})

    locations_csvs = data_client.request(
        request,
        fields='dot_locations'
    )['dot_locations']

    if locations_csvs:
        df = aggregate_dot_dfs(locations_csvs, hyb, position)

        return dbc.Table.from_dataframe(df, striped=True, size='sm')
    else:
        return []


@app.callback(
    Output('dd-detail-tabs-collapse', 'is_open'),
    Output('dd-chan-select', 'options'),
    Output('dd-image-params-loader', 'children'),
    Input('dd-position-select', 'value'),
    Input('dd-hyb-select', 'value'),
    Input('dataset-select', 'value'),
    Input('user-select', 'value')
)
def select_pos_hyb(position, hyb, dataset, user):

    logger.info('Entering select_pos_hyb')
    # Note we test for None rather than truthiness because a position or hyb of 0
    # evaluates to False when cast to bool, but is in fact a real value.
    is_open = not any([v is None for v in (position, hyb, dataset, user)])

    # Close the collapser and reset the components
    if not is_open:
        return (
            False,
            no_update,
            [
                *cm.component_group('image-params', tolist=True),
                html.Div([
                    cm.component('dd-strictness-slider')
                ], id='dd-strictness-slider-wrapper')
            ]
        )

    logger.info('select_pos_hyb: Requesting raw image filename')
    imagefile = data_client.request(
        {'user': user, 'dataset': dataset, 'hyb': hyb, 'position': position},
        fields='hyb_fov'
    )
    logger.info('select_pos_hyb: got raw image filename')

    try:
        image = pil_imread(imagefile['hyb_fov'][0])
        if image.ndim == 3:
            image = image[:, None, :]

    except (AssertionError, IndexError, RuntimeError) as e:
        print(e, type(e))
        return (
            True,
            no_update,
            [
                dbc.Alert(f'No image file for dataset {user}/{dataset} '
                          f'hyb {hyb} position {position} found!', color='warning'),

                *cm.component_group(
                    'image-params',
                    tolist=True,
                    options=dict(disabled=True)
                ),
                html.Div([
                    cm.component('dd-strictness-slider')
                ], id='dd-strictness-slider-wrapper')
            ]
        )

    logger.info('select_pos_hyb: read in raw image file')

    z_ind = 1
    c_ind = 0

    z_range = list(range(image.shape[z_ind]))
    chan_range = list(range(image.shape[c_ind]))

    marks = {a * 256: '{:0.1}'.format(a) for a in np.linspace(0, 1, 11)}

    logger.info('select_pos_hyb: returning image_params components')

    return (
        # is_open
        True,
        # dd-chan-select.options
        [{'label': str(c), 'value': str(c)} for c in chan_range],
        # image params children
        [
            html.B('Select Z slice'),
            cm.component(
                'dd-z-select',
                min=-1,
                max=z_range[-1],
                step=1,
                value=0,
                marks={-1: 'Max'} | {z: str(z) for z in z_range}
            ),

            html.B('Adjust contrast'),
            html.Div([
                cm.component(
                    'dd-contrast-slider',
                    min=0,
                    max=255,
                    step=1,
                    marks=marks,
                    value=[0, 200],
                    allowCross=False
                ),
                cm.component('dd-contrast-note')
            ], id='dd-contrast-div'),
            html.Div([
                cm.component('dd-strictness-slider')
            ], id='dd-strictness-slider-wrapper')
        ]
    )


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

    return [{'label': '(new)', 'value': '__new__'}] + \
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

    hybs = data_client.datafiles.query(
        'user == @user and dataset == @dataset '
        'and source_key == "hyb_fov"')['hyb'].dropna().unique()

    hybs_sorted = sort_as_num_or_str(hybs)

    positions = data_client.datafiles.query(
        'user == @user and dataset == @dataset '
        'and source_key == "hyb_fov"')['position'].dropna().unique()

    positions_sorted = sort_as_num_or_str(positions)

    return cm.component_group(
        'image-select',
        tolist=True,
        options=
        {
            'dd-hyb-select':
                dict(options=[{'label': h, 'value': h} for h in hybs_sorted]),
            'dd-position-select':
                dict(options=[{'label': p, 'value': p} for p in positions_sorted]),
        }
    )


@app.callback(
    [
     Output('dd-image-select-wrapper', 'is_open'),
     Output('dd-analysis-select', 'value'),
     ],
    [Input('dataset-select', 'value'),
     Input('user-select', 'value')
     ]
)
def reset_dependents(dataset, user):
    # Close the image-select wrapper if either is not defined
    image_select_open = dataset and user

    # Always reset the analysis value
    analysis_value = None

    return image_select_open, analysis_value


tab_dot_detection = dbc.Spinner([
    *cm.component_group(
        'image-params',
        tolist=True),
    html.Div(cm.component(
        'dd-strictness-slider',
    ), id='dd-strictness-slider-wrapper')
], id='dd-image-params-loader'),

tab_preprocess_checks = dbc.Alert(
    'Select different hyb and positions '
    'to see the preprocess check images at right.', color='success')

tab_alignment_check = dbc.Alert(
    'Use the slider to navigate through the DAPI alignment check stack.',
    color='success'
)

tab_locations_check = dbc.Alert(
    'Images of dot locations in XY or across Z slices.',
    color='success'
)

layout = [
    dbc.Col([
        html.Div([
            *cm.component_group('dataset-info', tolist=True),

            dbc.Collapse([
                *cm.component_group('image-select', tolist=True)
            ], is_open=False, id='dd-image-select-wrapper'),

        ], id='dd-dataset-select-div', style={'margin': '10px'}),

        html.Hr(),

        html.Details([
            html.Summary('Breakdown by position+channel of detected dots'),
            dbc.Card([
                dbc.CardHeader('Detected dots breakdown'),
                dbc.CardBody(
                    dbc.Spinner(html.Div(id='dd-dot-breakdown-wrapper'))
                ),
            ]),
        ]),

        dbc.Collapse([
            dbc.Tabs([
                dbc.Tab(tab_dot_detection,
                        tab_id='dd-tab-dotdetection',
                        label='Dot detection'),
                dbc.Tab(tab_preprocess_checks,
                        tab_id='dd-tab-preprocess',
                        label='Preprocessing Checks'),
                dbc.Tab(tab_alignment_check,
                        tab_id='dd-tab-alignment',
                        label='DAPI Alignment Check'),
                dbc.Tab(tab_locations_check,
                        tab_id='dd-tab-locations',
                        label='Dot Locations Check')
            ], id='dd-detail-tabs', style={'margin': '10px'}),
        ], is_open=False, id='dd-detail-tabs-collapse')

    ], width=4),

    dbc.Col([
        dcc.Loading(cm.component('dd-fig'), id='dd-graph-wrapper')
    ], id='dd-fig-col', width='auto')
]
