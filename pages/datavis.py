import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, config, s3_client
from util import populate_mesh, base64_image
from cloud import DatavisStorage


data_manager = DatavisStorage(config=config, s3_client=s3_client)
data_manager.get_datasets()


def _query_df(selected_genes, name):
    """
    _query_df:
    memoizable function (though see note below) that filters the active
    data dataframe for a given list of genes.
    
    returns: JSON representation (for caching) of filtered dot list
    
    Memoization note:
      This needs to assume that ACTIVE_DATA['dots'] is populated
      So we need to only call it in that case (see `gen_figure`)
      Also, we supply it with ACTIVE_DATA['name'] for proper memoization,
      as the cache is valid only for a specific combo of gene list + active dataset.
    """
    
    assert data_manager.active_dots is not None, '`query_df` assumes ACTIVE_DATA is populated.'
    
    if 'All' in selected_genes:
        dots_filt = data_manager.active_dots
    elif selected_genes == ['None']:
        dots_filt = data_manager.active_dots.query('gene == "NOT__A__GENE"')
    else:
        dots_filt = data_manager.active_dots.query('gene in @selected_genes')
    
    return dots_filt.to_json()


def query_df(selected_genes):
    """
    query_df:
    memoization helper that feeds `_query_df` the gene list and dataset name,
    and decodes from JSON into a DataFrame.
    
    returns: Filtered DataFrame
    """
    return pd.read_json(_query_df(selected_genes, data_manager.active_dataset_name))


#@cache.memoize()
def gen_figure(selected_genes, name):
    """
    gen_figure:
    Given a list of selected genes and a dataset, generates a Plotly figure with 
    Scatter3d and Mesh3d traces for dots and cells, respectively. Memoizes using the
    gene selection and active dataset name.
    
    If ACTIVE_DATA is not set, as it isn't at initialization, this function
    generates a figure with an empty Mesh3d and Scatter3d. 
    
    Returns: plotly.graph_objects.Figure containing the selected data.
    """
    
    figdata = []
    
    # If dots is populated, grab it.
    # Otherwise, set the coords to None to create an empty Scatter3d.
    if data_manager.active_dots is not None:
        dots_filt = query_df(selected_genes)
        
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
    if data_manager.active_mesh is not None:
        x, y, z, i, j, k = populate_mesh(data_manager.active_mesh)
    
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
    [Input('gene-select', 'value')],
    prevent_initial_call=False
)
def update_figure(selected_genes,):
    """
    update_figure:
    Callback triggered by by selecting
    gene(s) to display. Calls `gen_figure` to populate the figure on the page.
    
    """
    
    #### NOTE may need to use Dash's callback context to determine
    ## whether data-select or gene-select triggered this
    
    if (data_manager.active_dataset_name is not None 
       and selected_genes == []):
        raise PreventUpdate
        
    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]
        
    if 'All' in selected_genes:
        selected_genes = ['All']
    
    if 'None' in selected_genes and len(selected_genes) > 1:
        selected_genes.remove('None')
    
    fig = gen_figure(selected_genes, data_manager.active_dataset_name)
    
    #end = datetime.now()
    #print(f'returning from callback at {end} after {end-start}')
    
    return dcc.Graph(id='test-graph', figure=fig)


@app.callback(
    Output('analytics-wrapper', 'children'),
    Input('pos-select', 'value'),
    prevent_initial_call=True
)
def populate_analytics(pos):
    if data_manager.active_position_name != pos:
        active = data_manager.select_position(pos)
    else:
        active = data_manager.active_position

    file1 = active.get('onoff_int_file', None)
    data1 = base64_image(file1)
    file2 = active.get('onoff_sorted_file', None)
    data2 = base64_image(file2)
    print(f'file1: {file1}, data2: {file1}')

    print(f'data1: {data1[:50]}, data2: {data2[:50]}')

    return [html.Img(src=data1, style={'max-width': '100%'}),
            html.Hr(),
            html.Img(src=data2, style={'max-width': '100%'})
        ]

@app.callback(
    Output('gene-wrapper', 'children'),
    Input('pos-select', 'value')
)
def select_pos(pos):
    #....
    active = data_manager.select_position(pos)
    # TODO: separate dropdowns for each channel?
    all_genes = np.ravel(list(active['genes'].values()))
    
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
    [Input('data-select', 'value')]
)
def select_data(folder):

    if folder is None:
        return None
    
    dataset, _ = data_manager.select_dataset(folder)
    positions = list(dataset.keys())
    
    return [
        dcc.Dropdown(
            id='pos-select',
            options=[{'label': i, 'value': i} for i in positions],
            value=positions[0],
            placeholder='Select position',
            clearable=False,
            style={'width': '200px', 'margin': '20px'}),

            html.Div([dcc.Loading(dcc.Dropdown(id='gene-select'), id='gene-wrapper')],
                id='gene-div',
                style={'width': '200px', 'margin': '20px'})
    ]

######## Layout ########

layout = dbc.Container(dbc.Row([
    dbc.Col([
        html.Div([
            html.H2('Data selection'),
            html.Hr(),
            dcc.Dropdown(
                id='data-select',
                options=[{'label': i, 'value': i} for i in data_manager.datasets.keys()],
                placeholder='Select a data folder',
                style={'width': '200px'}
            ),
           ], id='selector-div'),
        
        dcc.Loading([
            html.Div(
                [dcc.Loading(dcc.Dropdown(id='pos-select'), id='pos-wrapper')],
                id='pos-div'),
            ], id='selectors-wrapper', style={'width': '200px', 'margin': '20px'}),

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
