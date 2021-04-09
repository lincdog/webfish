import os
import json
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask_caching import Cache

from app import app, cache, config, s3_client
from util import gen_mesh, gen_pcd_df, mesh_from_json, populate_mesh, populate_genes
from cloud import DatavisStorage

ACTIVE_DATA = {'name': None, 
               'position': None,
               'mesh': None, 
               'dots': None
              }
HAS_MESH = False


data_manager = DatavisStorage(config=config, s3_client=s3_client)
data_manager.get_datasets()


@cache.memoize()
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
    
    assert ACTIVE_DATA['dots'] is not None, '`query_df` assumes ACTIVE_DATA is populated.'
    
    if 'All' in selected_genes:
        dots_filt = ACTIVE_DATA['dots']
    elif selected_genes == ['None']:
        dots_filt = ACTIVE_DATA['dots'].query('geneID == "NOT__A__GENE"')
    else:
        dots_filt = ACTIVE_DATA['dots'].query('geneID in @selected_genes')
    
    return dots_filt.to_json()

def query_df(selected_genes):
    """
    query_df:
    memoization helper that feeds `_query_df` the gene list and dataset name,
    and decodes from JSON into a DataFrame.
    
    returns: Filtered DataFrame
    """
    return pd.read_json(_query_df(selected_genes, ACTIVE_DATA['name']))


@cache.memoize()
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
    if ACTIVE_DATA['dots'] is not None:
        dots_filt = query_df(selected_genes)
        
        pz,py,px = dots_filt[['z', 'y', 'x']].values.T
        
        color = dots_filt['geneColor']
        hovertext = dots_filt['geneID']
    
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
    if ACTIVE_DATA['mesh'] is not None:
        x, y, z, i, j, k = populate_mesh(ACTIVE_DATA['mesh'])
    
        figdata.append(
        go.Mesh3d(x=x, y=y, z=z,
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

    figlayout= go.Layout(
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
    [Input('gene-select', 'value'),
     #Input('pos-select', 'value')
    ],
    prevent_initial_call=False
)
def update_figure(selected_genes,): # selected_pos):
    """
    update_figure:
    Callback triggered by by selecting
    gene(s) to display. Calls `gen_figure` to populate the figure on the page.
    
    """
    
    #### NOTE may need to use Dash's callback context to determine
    ## whether data-select or gene-select triggered this
    
    #start = datetime.now()
    #print(f'starting callback at {start}, genes = {selected_genes} name = {ACTIVE_DATA["name"]}')
    
    if (ACTIVE_DATA['name'] is not None 
        and selected_genes == []
       ):
        print(f'reached, has mesh: {HAS_MESH}')
        if HAS_MESH:
            raise PreventUpdate
        else:
            HAS_MESH = True
            return gen_figure(None, ACTIVE_DATA['name'])
        
    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]
        
    if 'All' in selected_genes:
        selected_genes = ['All']
    
    if 'None' in selected_genes and len(selected_genes) > 1:
        selected_genes.remove('None')
    
    fig = gen_figure(selected_genes, ACTIVE_DATA['name'])
    
    #end = datetime.now()
    #print(f'returning from callback at {end} after {end-start}')
    
    return dcc.Graph(id='test-graph', figure=fig)


@app.callback(
    Output('gene-div', 'children'),
    Input('pos-select', 'value')
)
def select_pos(pos):
    #....
    ACTIVE_DATA['position'] = pos
    return []
    ACTIVE_DATA['mesh'] = mesh
    ACTIVE_DATA['dots'] = pcd
    return dcc.Dropdown(
            id='pos-select',
            options=[{'label': i, 'value':i} for i in genes],
            value='Pos0',
            placeholder='Select position'
        )

@app.callback(
    Output('pos-wrapper', 'children'),
    [Input('data-select', 'value')]
)
def select_data(folder):
    HAS_MESH = False
    
    if folder is None:
        return None
    
    dataset = data_manager.select_dataset(folder)
    positions = list(dataset.keys())
    ### Set global dots DF and mesh variables
    ACTIVE_DATA['name'] = folder
    
    return [
        dcc.Dropdown(
            id='pos-select',
            options=[{'label': i, 'value': i} for i in positions],
            value=positions[0],
            placeholder='Select position',
            style={}
        ),
        
    ]
    
    
######## Layout ########

layout = dbc.Row([
    dbc.Col([
        html.Div([
            dcc.Dropdown(
                id='data-select',
                options=[{'label': i, 'value': i} for i in data_manager.datasets.keys()],
                placeholder='Select a data folder',
                style={'width': '200px', 'margin': 'auto'}
            ),
           ], id='selector-div', style={}),
        
        html.Div([dcc.Loading(dcc.Dropdown(id='pos-select'), id='pos-wrapper')],
             id='pos-div', style={'width': '200px', 'margin': 'auto'}),
        html.Div([dcc.Dropdown(id='gene-select')],
             id='gene-div', style={'width': '200px', 'margin': 'auto'})
    ], width=4),

    dbc.Col([
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id='test-graph',
                ), id='graph-wrapper'
            )
        ])
    ], width="auto")
])