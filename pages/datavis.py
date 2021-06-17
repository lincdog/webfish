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
from lib.util import (
    populate_mesh,
    base64_image,
    populate_genes,
    mesh_from_json,
    safe_imread
)

from .common import ComponentManager, data_client


clear_components = {
    'dv-gene-select': None,
    'dv-vis-mode': None,
    'dv-color-option': None,
    'dv-analysis-select': None,
    'dv-pos-select': None,
    'dv-fig': None
}

logger = logging.getLogger('webfish.' + __name__)


def query_df(df, selected_genes):
    """
    query_df:

    returns: Filtered DataFrame
    """
    if 'All' in selected_genes:
        return df
    elif 'All Real' in selected_genes:
        selected_genes.extend([gene for gene in df['gene'] if 'fake' not in gene])
    elif 'All Fake' in selected_genes:
        selected_genes.extend([gene for gene in df['gene'] if 'fake' in gene])

    return df.query('gene in @selected_genes')


def gen_figure_2d(selected_genes, active, color_option, channel):

    dots = active.get('dots')

    logger.info('Entering gen_figure_2d')

    fig = go.Figure()

    if 'background_im' in active:
        imfile = active.get('background_im')
        imtype = 'background_im'
    elif 'presegmentation_im' in active:
        imfile = active.get('presegmentation_im')
        imtype = 'presegmentation_im'
    else:
        imfile = None
        imtype = ''

    if imfile:
        img = safe_imread(imfile[0])

        logger.info('gen_figure_2d: read in 2d image')

        # TODO: Allow choosing Z slice?
        # FIXME: Choose channel or use channel that was used in decoding
        if img.ndim == 4:
            img = np.max(img[channel], axis=0)
        elif img.ndim == 3:
            img = img[channel]

        fig = px.imshow(
            img,
            zmin=0,
            zmax=200,
            width=1000,
            height=1000,
            binary_string=True
        )

        logger.info('gen_figure_2d: created image trace')

    # If dots is populated, grab it.
    # Otherwise, set the coords to None to create an empty Scatter3d.
    if dots is not None:

        dots_df = pd.read_csv(dots)
        dots_filt = query_df(dots_df, selected_genes).copy()
        del dots_df

        logger.info('gen_figure_2d: read and queried dots DF')

        py, p_x = dots_filt[['y', 'x']].values.T

        color = dots_filt['geneColor']
        if color_option == 'fake':
            real_fake = ('cyan', 'magenta')
            color = [real_fake[int('fake' in g)] for g in dots_filt['gene']]

        hovertext = dots_filt['gene']

        fig.add_trace(
            go.Scattergl(
                name='dots',
                x=p_x, y=py,
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=1,
                    symbol='cross',
                ),
                hoverinfo='x+y+text',
                hovertext=hovertext
            )
        )

        logger.info('gen_figure_2d: created Scattergl trace')

    logger.info('gen_figure_2d: returning updated figure')

    return fig


def gen_figure_3d(selected_genes, active, color_option, z_step_size, pixel_size):
    """
    gen_figure_3d:
    Given a list of selected genes and a dataset, generates a Plotly figure with
    Scatter3d and Mesh3d traces for dots and cells, respectively. Memoizes using the
    gene selection and active dataset name.

    If ACTIVE_DATA is not set, as it isn't at initialization, this function
    generates a figure with an empty Mesh3d and Scatter3d.

    Returns: plotly.graph_objects.Figure containing the selected data.
    """
    logger.info('Entering gen_figure_3d')

    print(active)
    dots = active.get('dots')
    mesh = active.get('mesh')

    figdata = []

    # If dots is populated, grab it.
    # Otherwise, set the coords to None to create an empty Scatter3d.
    if dots is not None:

        dots_df = pd.read_csv(dots)
        dots_filt = query_df(dots_df, selected_genes).copy()
        del dots_df

        logger.info('gen_figure_3d: read and queried dots DF')

        pz, py, p_x = dots_filt[['z', 'y', 'x']].values.T

        color = dots_filt['geneColor']
        if color_option == 'fake':
            real_fake = ('#1d4', '#22a')
            color = [real_fake[int('fake' in g)] for g in dots_filt['gene']]

        hovertext = dots_filt['gene']

        figdata.append(
            go.Scatter3d(
                name='dots',
                x=p_x, y=py, z=pz,
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=1,
                    symbol='circle',
                ),
                hoverinfo='text',
                hovertext=hovertext
            )
        )

        logger.info('gen_figure_3d: added Scatter3d trace')

    # A sensible default for aesthetic purposes (refers to the ratio between the
    # total extent in the Z dimension to that in the X or Y direction
    z_aspect = 0.07
    # If the mesh is present, populate it.
    # Else, create an empty Mesh3d.
    if mesh is not None:

        x, y, z, i, j, k = populate_mesh(mesh_from_json(mesh))

        figdata.append(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='lightgray',
                opacity=0.6,
                hoverinfo='skip',
            )
        )

        logger.info('gen_figure_3d: Added mesh3d trace')

        if pixel_size and z_step_size:
            x_extent = pixel_size * (x.max() - x.min())
            z_extent = z_step_size * (z.max() - z.min())

            z_aspect = z_extent / x_extent

            logger.info(f'gen_figure_3d: px {pixel_size} z {z_step_size} '
                        f'gives x extent {x_extent} z extent {z_extent} '
                        f'ratio = {z_aspect}')

    figscene = go.layout.Scene(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=z_aspect),
    )

    figlayout = go.Layout(
        height=1000,
        width=1000,
        margin=dict(b=10, l=10, r=10, t=10),
        scene=figscene
    )

    fig = go.Figure(data=figdata, layout=figlayout)

    logger.info('gen_figure_3d: returning figure')

    return fig


####### Callbacks #######

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

    active = data_client.request({
        'user': user,
        'dataset': dataset,
        'analysis': analysis,
        'position': pos
    }, fields='dots')

    logger.info('update_figure: got mesh and dots files')

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

        fig = gen_figure_3d(
            selected_genes,
            active,
            color_option,
            z_step_size,
            pixel_size
        )

    else:
        if source_2d == 'dapi_im':
            source_2d = 'background_im'
            channel = -1
        else:
            channel = 0

        active |= data_client.request({
            'user': user,
            'dataset': dataset,
            'position': pos
        }, fields=source_2d)

        config = {'scrollZoom': True}

        if not active[source_2d]:
            info = dbc.Alert('No 2D source image of the requested type',
                             color='warning')

        if (not active[source_2d]) and (not active['dots']):
            return [dbc.Alert('Source image and dots not found!', color='warning'),
                    dcc.Graph(id='dv-fig')]

        fig = gen_figure_2d(selected_genes, active, color_option, channel)

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
        'user==@user and dataset==@dataset and analysis==@analysis')['position'].unique()

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
            clearable=False
        )
    ]


######## Layout ########


layout = [
    dbc.Col([
        html.Div([
            html.H2('Data selection'),
            html.Div([
                dcc.Dropdown(id='dv-analysis-select')
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
                            {'label': '2D (multi position)', 'value': '2d'}
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
