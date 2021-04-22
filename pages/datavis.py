import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, config, s3_client
from util import populate_mesh, base64_image, populate_genes, mesh_from_json
from cloud import DatavisStorage


data_manager = DatavisStorage(config=config, s3_client=s3_client)
datasets = data_manager.get_datasets()


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
    Output('graph-wrapper', 'children'),
    Input('gene-select', 'value'),
    State('data-select', 'value'),
    State('pos-select', 'value'),
    prevent_initial_call=False
)
def update_figure(selected_genes, dataset, pos):
    """
    update_figure:
    Callback triggered by by selecting
    gene(s) to display. Calls `gen_figure` to populate the figure on the page.

    """

    if not all((dataset, pos)):
        raise PreventUpdate

    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]

    if 'All' in selected_genes:
        selected_genes = ['All']

    active = data_manager.request({
        'dataset': dataset,
        'position': pos
    }, fields=['mesh', 'dots'])

    fig = gen_figure(selected_genes, active)

    return dcc.Graph(id='test-graph', figure=fig)


@app.callback(
    Output('analytics-wrapper', 'children'),
    Input('pos-select', 'value'),
    State('data-select', 'value'),
    prevent_initial_call=True
)
def populate_analytics(pos, dataset):

    if not pos:
        raise PreventUpdate

    active = data_manager.request({
        'dataset': dataset,
        'position': pos
    }, fields=['onoff_intensity_plot', 'onoff_sorted_plot'])

    data1 = base64_image(active['onoff_intensity_plot'][0])
    data2 = base64_image(active['onoff_sorted_plot'][0])

    return [html.Img(src=data1, style={'max-width': '100%'}),
            html.Hr(),
            html.Img(src=data2, style={'max-width': '100%'})
        ]


@app.callback(
    Output('gene-wrapper', 'children'),
    Input('pos-select', 'value'),
    State('data-select', 'value')
)
def select_pos(pos, dataset):
    if not pos:
        raise PreventUpdate

    active = data_manager.request({
        'dataset': dataset,
        'position': pos
    }, fields=['dots'])
    # TODO: separate dropdowns for each channel?

    dots = pd.read_csv(active['dots'])

    all_genes = populate_genes(dots)

    return [
        dcc.Dropdown(
            id='gene-select',
            options=[{'label': i, 'value': i} for i in all_genes],
            multi=True,
            placeholder='Select gene(s)'
        )
    ]


@app.callback(
    Output('selectors-wrapper', 'children'),
    Input('data-select', 'value')
)
def select_data(folder):

    if not folder:
        raise PreventUpdate

    print(folder)
    print(f'dataset == "{folder}"')
    rel_files = datasets.query(f'dataset == "{folder}"')

    positions = sorted(rel_files['position'].unique())
    print(rel_files)

    return [
        'Position select: ',
        dcc.Dropdown(
            id='pos-select',
            options=[{'label': i, 'value': i} for i in positions],
            value=positions[0],
            placeholder='Select position',
            clearable=False,
            style={'width': '200px',}),
        'Gene select: ',
            html.Div([dcc.Loading(dcc.Dropdown(id='gene-select'), id='gene-wrapper')],
                id='gene-div',
                style={'width': '200px',})
    ]


@app.callback(
    Output('wf-store', 'data'),
    Input('data-select', 'value'),
    Input('pos-select', 'value'),
    Input('gene-select', 'value'),
    State('wf-store', 'data'),
    prevent_initial_call=True
)
def store_manager(folder, pos, selected_genes, store):
    store = store or {'datavis-dataset': None,
                      'datavis-position': None,
                      'datavis-selected-genes': None
                      }

    print(f'store_manager called with store = {store}')
    print(f'folder: {folder} pos: {pos} selected_genes: {selected_genes}')

    if folder != store['datavis-dataset']:
        store['datavis-dataset'] = folder

    if pos != store['datavis-position']:
        store['datavis-position'] = pos

    if selected_genes != store['datavis-selected-genes']:
        store['datavis-selected-genes'] = selected_genes

    return store

######## Layout ########


layout = dbc.Container(dbc.Row([
    dbc.Col([
        html.Div([
            html.H2('Data selection'),
            html.Hr(),
            'Dataset select:',
            dcc.Dropdown(
                id='data-select',
                options=[{'label': i, 'value': i} for i in datasets['dataset'].unique()],
                placeholder='Select a data folder',
                style={'width': '200px'}
            ),
           ], id='selector-div'),

        dcc.Loading([
            html.Div(
                dcc.Loading(dcc.Dropdown(id='pos-select'),
                    id='pos-wrapper'),
                id='pos-div'),
            html.Div(
                dcc.Loading(dcc.Dropdown(id='gene-select'),
                    id='gene-wrapper'),
                id='gene-div')
            ], id='selectors-wrapper', style={'width': '200px', 'margin': '20px'}),
        html.Hr(),

        html.Div([
            html.H2('Analytics'),
            html.Hr(),
            dcc.Loading([

            ], id='analytics-wrapper')
        ])

    ], width=4, style={'border-right': '1px solid gray'}),

    dbc.Col([
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id='test-graph',
                ), id='graph-wrapper'
            )
        ])
    ], width="auto")
]), fluid=True)
