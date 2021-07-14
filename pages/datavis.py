import dash
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate

from app import app

from pages._page_util import (
    DatavisHelper,
    base64_image,
    populate_genes,
)
from pages.common import ComponentManager, data_client


clear_components = {
    'dv-gene-select': None,
    'dv-vis-mode': None,
    'dv-color-option': None,
    'dv-analysis-select': None,
    'dv-pos-select': None,
    'dv-fig': None
}

logger = logging.getLogger('webfish.' + __name__)
helper = DatavisHelper(data_client, ComponentManager({}), logger=logger)


@app.callback(
    Output('dv-graph-wrapper', 'children'),
    Input('dv-gene-select', 'value'),
    Input('dv-color-option', 'value'),
    Input('dv-vis-mode', 'value'),
    Input('dv-2d-source', 'value'),
    Input('dv-z-step-size', 'value'),
    Input('dv-pixel-size', 'value'),
    State('dv-pos-select', 'value'),
    State('dv-analysis-select', 'value'),
    State('dataset-select', 'value'),
    State('user-select', 'value'),
    State('dv-fig', 'relayoutData'),
    prevent_initial_call=True
)
def update_figure(
    selected_genes,
    color_option,
    vis_mode,
    source_2d,
    z_step_size,
    pixel_size,
    pos,
    analysis,
    dataset,
    user,
    current_layout
):
    """
    update_figure:
    Callback triggered by by selecting
    gene(s) to display. Calls `gen_figure_3d` to populate the figure on the page.

    """

    logger.info('Starting update_figure')

    if not all((user, dataset, analysis, pos)):
        raise PreventUpdate

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'dv-2d-source' and vis_mode != '2d':
        raise PreventUpdate

    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]

    if 'All' in selected_genes or (
            'All Real' in selected_genes and 'All Fake' in selected_genes):
        selected_genes = ['All']

    logger.info('update_figure: requesting mesh and dots files')

    info = None
    active = {}

    if vis_mode == 'all':
        active |= data_client.request({
            'user': user,
            'dataset': dataset,
            'analysis': analysis
        }, fields=[
            'allpos_falsepositive_png',
            'allpos_genecorr_clustered',
            'allpos_genecorr_unclustered',
            'allpos_onoff_plot',
            'allpos_percentagedots_png',
            'allpos_genespercell_png'
        ])
    else:
        active |= data_client.request({
            'user': user,
            'dataset': dataset,
            'analysis': analysis,
            'position': pos
        }, fields='dots')

        logger.info('update_figure: got dots file')

    config = {}

    if vis_mode == '3d':
        active |= data_client.request({
            'user': user,
            'dataset': dataset,
            'analysis': analysis,
            'position': pos
        }, fields='mesh')

        if not active['mesh']:
            info = dbc.Alert('No segmented cell image found', color='warning')

        if (not active['mesh']) and (not active['dots']):
            return [dbc.Alert('Segmented image and dots not found!', color='warning'),
                    dcc.Graph(id='dv-fig')]

        fig = helper.gen_figure_3d(
            selected_genes,
            active,
            color_option,
            z_step_size,
            pixel_size
        )

    elif vis_mode == '2d':
        source_2d_request = {
            'user': user,
            'dataset': dataset,
            'position': pos
        }

        if source_2d == 'dapi_im':
            source_2d = 'hyb_fov'

            first_hyb = data_client.datafiles.query(
                'user==@user and dataset==@dataset and position==@pos'
            )['hyb'].dropna().values[0]

            source_2d_request['hyb'] = first_hyb

            channel = -1
        else:
            channel = 0

        logger.info(f'source_2d_request: {source_2d_request}, {source_2d}')

        active |= data_client.request(source_2d_request, fields=source_2d)

        config = {'scrollZoom': True}

        if not active[source_2d]:
            info = dbc.Alert('No 2D source image of the requested type',
                             color='warning')

        if (not active[source_2d]) and (not active['dots']):
            return [dbc.Alert('Source image and dots not found!', color='warning'),
                    dcc.Graph(id='dv-fig')]

        fig = helper.gen_figure_2d(selected_genes, active, color_option, channel)

    elif vis_mode == 'all':
        logger.info(f'Generating all position analytics figure')

        current_layout = None

        fig = helper.gen_figure_allpos(active)

    if current_layout:
        if 'scene.camera' in current_layout:
            fig['layout']['scene']['camera'] = current_layout['scene.camera']

        if 'xaxis.range[0]' in current_layout:
            fig['layout']['xaxis']['range'] = [
                current_layout['xaxis.range[0]'],
                current_layout['xaxis.range[1]']
            ]
        if 'yaxis.range[0]' in current_layout:
            fig['layout']['yaxis']['range'] = [
                current_layout['yaxis.range[0]'],
                current_layout['yaxis.range[1]']
            ]

    logger.info('update_figure: returning constructed figure')

    return [info, dcc.Graph(id='dv-fig',
                            figure=fig,
                            relayoutData=current_layout,
                            config=config)]


@app.callback(
    Output('dv-analytics-wrapper', 'children'),
    Input('dv-pos-select', 'value'),
    State('dv-analysis-select', 'value'),
    State('dataset-select', 'value'),
    State('user-select', 'value'),
    prevent_initial_call=True
)
def populate_analytics(pos, analysis, dataset, user):
    logger.info('Entering populate_analytics')

    if not pos:
        raise PreventUpdate

    active = data_client.request({
        'user': user,
        'dataset': dataset,
        'analysis': analysis,
        'position': pos
    }, fields=[
        'onoff_intensity_plot',
        'onoff_sorted_plot',
        'falsepositive_txt',
        'genes_assigned_to_cells'
    ])

    if not active['onoff_intensity_plot']:
        img1 = html.B('On/off target intensity plot not found!')
    else:
        img1 = html.Img(src=base64_image(active['onoff_intensity_plot'][0]),
                        style={'max-width': '100%'})

    if not active['onoff_sorted_plot']:
        img2 = html.B('On/off target sorted barcode plot not found!')
    else:
        img2 = html.Img(src=base64_image(active['onoff_sorted_plot'][0]),
                        style={'max-width': '100%'})

    if not active['genes_assigned_to_cells']:
        img3 = [html.B('Genes assigned to cells plot not found!')]
    else:
        img3 = [html.H5('Genes Assigned to Cells Plot'),
                html.Img(src=base64_image(active['genes_assigned_to_cells'][0]),
                        style={'max-width': '100%'})
        ]

    if not active['falsepositive_txt']:
        fp_comp = html.B('No false positive rate analysis found!')
    else:
        fp_txt = html.Pre(open(active['falsepositive_txt'][0], 'r').read())
        fp_comp = html.Div(
            [
                dbc.Button(
                    'Toggle false positive analysis',
                    id='dv-collapse-button',
                    color='primary',
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody(fp_txt)),
                    id='dv-collapse',
                ),
            ]
        )

    logger.info('Returning from populate_analytics')

    return [fp_comp,
            html.Hr(),
            img1,
            html.Hr(),
            img2,
            html.Hr(),
            *img3]


@app.callback(
    Output('dv-collapse', 'is_open'),
    [Input('dv-collapse-button', 'n_clicks')],
    [State('dv-collapse', 'is_open')],
    prevent_initial_call=True
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output('dv-gene-wrapper', 'children'),
    Input('dv-pos-select', 'value'),
    State('dv-analysis-select', 'value'),
    State('dataset-select', 'value'),
    State('user-select', 'value'),
    prevent_initial_call=True
)
def select_pos(pos, analysis, dataset, user):
    if not pos:
        raise PreventUpdate

    active = data_client.request({
        'user': user,
        'dataset': dataset,
        'analysis': analysis,
        'position': pos
    }, fields=['dots'])
    # TODO: separate dropdowns for each channel?

    if not active['dots']:
        return [html.B('Dot file not found!'),
                dcc.Dropdown(
                    id='dv-gene-select',
                    disabled=True,
                    multi=True,
                    placeholder='Select gene(s)'
                )
                ]

    dots = pd.read_csv(active['dots'])

    all_genes = populate_genes(dots)

    return [
        dcc.Dropdown(
            id='dv-gene-select',
            options=[{'label': i, 'value': i} for i in all_genes],
            multi=True,
            placeholder='Select gene(s)'
        )
    ]


@app.callback(
    Output('dv-pos-div', 'children'),
    Input('dv-analysis-select', 'value'),
    State('dataset-select', 'value'),
    State('user-select', 'value')
)
def select_analysis(analysis, dataset, user):
    if not analysis:
        raise PreventUpdate

    positions = data_client.datafiles.query(
        'user==@user and dataset==@dataset and analysis==@analysis')['position'].dropna().unique()

    positions_sorted = np.sort(positions.astype(int)).astype(str)

    return [
        'Position select: ',
        dcc.Dropdown(
            id='dv-pos-select',
            options=[{'label': i, 'value': i} for i in positions_sorted],
            value=positions[0],
            placeholder='Select a position',
            clearable=False
        )
    ]


@app.callback(
    Output('dv-analysis-select-div', 'children'),
    Input('dataset-select', 'value'),
    State('user-select', 'value')
)
def select_dataset(dataset, user):

    if not dataset:
        raise PreventUpdate

    analyses = data_client.datasets.query(
        'user==@user and dataset==@dataset')['analysis'].dropna().unique()
    print(analyses)

    return [
        'Analysis select: ',
        dcc.Dropdown(
            id='dv-analysis-select',
            options=[{'label': i, 'value': i} for i in sorted(analyses)],
            value=None,
            placeholder='Select an analysis run',
            clearable=False,
            persistence=True,
            persistence_type='session'
        )
    ]


######## Layout ########


layout = [
    dbc.Col([
        html.Div([
            html.H2('Data selection'),
            html.Div([
                dcc.Dropdown(
                    id='dv-analysis-select',
                    persistence=True,
                    persistence_type='session'
                )
            ], id='dv-analysis-select-div')
        ], id='dv-analysis-selectors-div'),

        html.Hr(),
        html.Div([

            html.Div(
                dcc.Loading(dcc.Dropdown(id='dv-pos-select', disabled=True),
                    id='dv-pos-wrapper'),
                id='dv-pos-div'
            ),

            html.Div([
                dcc.Loading(
                    dcc.Dropdown(id='dv-gene-select', disabled=True),
                    id='dv-gene-wrapper'
                ),

            ], id='dv-gene-div'),

            dbc.FormGroup([
                    dbc.Label('Visualization mode', html_for='dv-vis-mode'),
                    dbc.RadioItems(
                        id='dv-vis-mode',
                        options=[
                            {'label': '3D (one position)', 'value': '3d'},
                            {'label': '2D (multi position)', 'value': '2d'},
                            {'label': 'All position analytics', 'value': 'all'}
                        ],
                        value='2d',
                        inline=True
                    )
                ]),

            dbc.FormGroup([
                    dbc.Label('Color genes by...', html_for='dv-color-option'),
                    dbc.RadioItems(
                        id='dv-color-option',
                        options=[
                            {'label': 'Gene', 'value': 'gene'},
                            {'label': 'On/Off Target', 'value': 'fake'}
                        ],
                        value='gene',
                        inline=True
                    )
                ]),

            html.Details([
                html.Summary('3D visualization options'),
                html.Small(html.I('Note: changing these only changes the 3D plot'
                                  ' aspect ratio')),
                dbc.Form([
                    dbc.FormGroup([
                        dbc.Label('Z step size (microns):', html_for='dv-z-step-size'),
                        dbc.Input(
                            id='dv-z-step-size',
                            type='number',
                            min=0.1,
                            max=50,
                            step=0.1,
                            value=0.5,
                            debounce=True,
                            style={'width': '80px', 'margin': '10px'}
                        ),
                    ]),
                    dbc.FormGroup([
                        dbc.Label('XY pixel size (microns):', html_for='dv-pixel-size'),
                        dbc.Input(
                            id='dv-pixel-size',
                            type='number',
                            min=0.05,
                            max=0.5,
                            step=0.01,
                            value=0.11,
                            debounce=True,
                            style={'width': '80px', 'margin': '10px'}
                        ),
                    ]),
                ], inline=True)
            ]),

            html.Details([
                html.Summary('2D visualization options'),
                dbc.FormGroup([
                    dbc.Label('2D visualization source', html_for='dv-2d-source'),
                    dbc.RadioItems(
                        id='dv-2d-source',
                        options=[
                                {'label': 'Final background image',
                                 'value': 'background_im'},
                                {'label': 'Segmentation stain image',
                                 'value': 'presegmentation_im'},
                                {'label': 'DAPI Max projection',
                                 'value': 'dapi_im'}
                            ],
                        value='background_im',
                        inline=True
                    )
                ])
            ])
        ], id='dv-selectors-wrapper', style={'margin': '20px'}),

        html.Hr(),

        html.Div([
            html.H2('Analytics'),
            html.Hr(),
            dcc.Loading([

            ], id='dv-analytics-wrapper')
        ])

    ], width=4),

    dbc.Col([
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id='dv-fig',
                ), id='dv-graph-wrapper'
            )
        ])
    ], width='auto')
]
