import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate

from app import app
from lib.util import populate_mesh, base64_image, populate_genes, mesh_from_json
from .common import data_client


def query_df(df, selected_genes):
    """
    query_df:

    returns: Filtered DataFrame
    """
    if 'All' in selected_genes:
        return df

    return df.query('gene in @selected_genes')


def gen_figure(selected_genes, active):
    """
    gen_figure:
    Given a list of selected genes and a dataset, generates a Plotly figure with
    Scatter3d and Mesh3d traces for dots and cells, respectively. Memoizes using the
    gene selection and active dataset name.

    If ACTIVE_DATA is not set, as it isn't at initialization, this function
    generates a figure with an empty Mesh3d and Scatter3d.

    Returns: plotly.graph_objects.Figure containing the selected data.
    """

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

        pz, py, px = dots_filt[['z', 'y', 'x']].values.T

        color = dots_filt['geneColor']
        hovertext = dots_filt['gene']

        figdata.append(
            go.Scatter3d(
                name='dots',
                x=px, y=py, z=pz,
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

    # If the mesh is present, populate it.
    # Else, create an empty Mesh3d.
    if mesh is not None:

        x, y, z, i, j, k = populate_mesh(mesh_from_json(mesh))

        figdata.append(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='lightgray',
                opacity=0.7,
                hoverinfo='skip',
            )
        )

    figscene = go.layout.Scene(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.07),
    )

    figlayout = go.Layout(
        height=800,
        width=800,
        #plot_bgcolor='black',
        #paper_bgcolor='white',
        margin=dict(b=10, l=10, r=10, t=10),
        scene=figscene
    )

    fig = go.Figure(data=figdata, layout=figlayout)

    return fig


####### Callbacks #######

@app.callback(
    Output('dv-graph-wrapper', 'children'),
    Input('dv-gene-select', 'value'),
    State('dv-pos-select', 'value'),
    State('dv-analysis-select', 'value'),
    State('dataset-select', 'value'),
    State('user-select', 'value'),
    State('dv-fig', 'relayoutData'),
    prevent_initial_call=True
)
def update_figure(
    selected_genes,
    pos,
    analysis,
    dataset,
    user,
    current_layout
):
    """
    update_figure:
    Callback triggered by by selecting
    gene(s) to display. Calls `gen_figure` to populate the figure on the page.

    """

    if not all((user, dataset, analysis, pos)):
        raise PreventUpdate

    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]

    if 'All' in selected_genes:
        selected_genes = ['All']

    active = data_client.request({
        'user': user,
        'dataset': dataset,
        'analysis': analysis,
        'position': pos
    }, fields=('mesh', 'dots'))

    if not active['mesh'] and not active['dots']:
        return html.H2('Segmented image and dots not found!')

    if not active['mesh']:
        info = dbc.Alert('Note: no segmented cell image found', color='warning')
    else:
        info = None

    print(current_layout.keys())

    fig = gen_figure(selected_genes, active)

    if current_layout:
        if 'scene.camera' in current_layout:
            fig['layout']['scene']['camera'] = current_layout['scene.camera']

    return [info, dcc.Graph(id='dv-fig', figure=fig, relayoutData=current_layout)]


@app.callback(
    Output('dv-analytics-wrapper', 'children'),
    Input('dv-pos-select', 'value'),
    State('dv-analysis-select', 'value'),
    State('dataset-select', 'value'),
    State('user-select', 'value'),
    prevent_initial_call=True
)
def populate_analytics(pos, analysis, dataset, user):

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
            clearable=False,
            style={'width': '200px'}
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
            html.Div(id='dv-analysis-select-div')
        ], id='dv-analysis-selectors-div'),

        html.Hr(),
        html.Div([
            html.Div(
                dcc.Loading(dcc.Dropdown(id='dv-pos-select', disabled=True),
                    id='dv-pos-wrapper'),
                id='dv-pos-div'),
            html.Div(
                dcc.Loading(dcc.Dropdown(id='dv-gene-select', disabled=True),
                    id='dv-gene-wrapper'),
                id='dv-gene-div')
            ], id='dv-selectors-wrapper', style={'width': '200px', 'margin': '20px'}),
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
