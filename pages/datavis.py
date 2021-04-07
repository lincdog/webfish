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
#from cloud import grab_bucket, download_s3_folder


ACTIVE_DATA = {'name': None, 'mesh': None, 'dots': None}
HAS_MESH = False

_, possible_folders = s3_client.grab_bucket( 
    config['bucket_name'],
    delimiter='/',
    prefix='',
    recursive=False
)


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
    
    start = datetime.now()
    print(f'starting callback at {start}, genes = {selected_genes} name = {ACTIVE_DATA["name"]}')
    
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
    
    end = datetime.now()
    print(f'returning from callback at {end} after {end-start}')
    
    return dcc.Graph(id='test-graph', figure=fig)


@app.callback(
    Output('gene-div', 'children'),
    [Input('data-select', 'value')]
)
def select_data(folder):
    HAS_MESH = False
    
    if folder is None:
        return None
    
    if not os.path.exists(config['local_store']):
        os.mkdir(config['local_store'])
        
    local_folder = os.path.join(config['local_store'], folder)
    
    def findlocal(name, local_folder=local_folder):
        return os.path.join(local_folder, name)
    
    # the desired folder doesn't exist, so we must fetch from s3
    if not os.path.exists(local_folder):
        s3_client.download_s3_objects(
            config['bucket_name'], 
            folder, 
            local_dir=config['local_store'],
            delimiter='/',
            recursive=False
        )
        
    assert os.path.exists(local_folder), f'Unable to fetch folder {folder} from s3.'
    
    print(f'{findlocal(config["mesh_name"])}, {findlocal(config["pcd_name"])}')
    
    # If we already have the mesh file for this, read from json.
    # else, generate it and save it.
    if os.path.exists(findlocal(config['mesh_name'])):
        mesh = mesh_from_json(findlocal(config['mesh_name']))
    else:
        mesh = mesh_from_json(
            gen_mesh(findlocal(config['img_name']),
                    outfile=findlocal(config['mesh_name'])))
    
    # if we already have the processed point cloud DF for this, read it in.
    # else, generate it and save it.
    if os.path.exists(findlocal(config['pcd_name'])):
        pcd = pd.read_csv(findlocal(config['pcd_name']))
    else:
        pcd = gen_pcd_df(findlocal(config['csv_name']), 
                         outfile=findlocal(config['pcd_name']))
    
    ### Set global dots DF and mesh variables
    ACTIVE_DATA['name'] = folder
    ACTIVE_DATA['mesh'] = mesh
    ACTIVE_DATA['dots'] = pcd
    
    
    genes = populate_genes(pcd)
    print('returning gene list')
    
    return [
        dcc.Loading(dcc.Dropdown(
            id='gene-select',
            options=[{'label': i, 'value': i} for i in genes],
            value='None',
            multi=True,
            placeholder='Select gene(s)',
            style={}
        )),
        #dcc.Dropdown(
        #    id='pos-select',
        #    options=[{'label': i, 'value':i} for i in positions],
        #    value='Pos0',
        #    placeholder='Select position'
        #)
    ]
    
    
######## Layout ########

layout = dbc.Row([
    dbc.Col([
        html.Div([
            dcc.Dropdown(
                id='data-select',
                options=[{'label': i, 'value': i} for i in possible_folders],
                placeholder='Select a data folder',
                style={'width': '200px', 'margin': 'auto'}
            ),
           ], id='selector-div', style={}),
        
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