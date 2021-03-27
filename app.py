import json
import os
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask_caching import Cache
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import boto3
import yaml

from util import gen_mesh, gen_pcd_df, mesh_from_json, populate_mesh, populate_genes

####### Globals #######

ACTIVE_DATA = {'name': None, 'mesh': None, 'dots': None}

config_file = 'consts.yml'
config = yaml.load(open(config_file), Loader=yaml.Loader)

###### AWS Code #######
# assumes credentials & configuration are handled outside python in .aws directory or environment variables

try:
    # Find the name of the credential file from the environment
    cred_file = os.environ[config['credentials']]
    
    import configparser as cfparse
    
    cf = cfparse.ConfigParser()
    cf.read(cred_file)
    
    # Find the desired profile section
    if 'cred_profile_name' in config.keys():
        csec = cf[config['cred_profile_name']]
    else:
        csec = cf['default']
    
    # Get the key ID and secret key
    key_id = csec['aws_access_key_id']
    secret_key = csec['aws_secret_access_key']

except:
    key_id = None
    secret_key = None
    
if 'endpoint_url' in config.keys():
    endpoint_url = config['endpoint_url']
else:
    endpoint_url = None
    

s3 = boto3.resource('s3', 
                    endpoint_url=endpoint_url,
                    aws_access_key_id=key_id,
                    aws_secret_access_key=secret_key
                   )

#BUCKET_NAME = 'lincoln-testing'
my_bucket = s3.Bucket(config['bucket_name'])

objects = list(my_bucket.objects.all())
# Unique folders sorted alphabetically
possible_folders = sorted(list(set([o.key.split('/')[0] for o in objects ])))

def download_s3_folder(s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
    bucket_name: the name of the s3 bucket
    s3_folder: the folder path in the s3 bucket
    local_dir: a relative or absolute directory path in the local file system
    """
    #bucket = s3.Bucket(bucket_name)
    
    for obj in my_bucket.objects.filter(Prefix=s3_folder):
        if local_dir is None:
            target = obj.key 
        else:
            target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        if obj.key.endswith('/'):
            continue
            
        my_bucket.download_file(obj.key, target)
        

############# Begin app code ############

THEME = dbc.themes.MINTY

app = dash.Dash(__name__, external_stylesheets=[THEME])

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

#### Helper functions ####

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
    Output('test-graph', 'figure'),
    Input('gene-select', 'value'),
    Input('data-select', 'value'),
    prevent_initial_call=False
    )
def update_figure(selected_genes, selected_data):
    """
    update_figure:
    Callback triggered by both selecting a dataset and by selecting
    gene(s) to display. Calls `gen_figure` to populate the figure on the page.
    
    """
    
    #### NOTE may need to use Dash's callback context to determine
    ## whether data-select or gene-select triggered this
    
    start = datetime.now()
    print(f'starting callback at {start}')
    
    if ACTIVE_DATA['name'] is not None and len(selected_genes) == 0:
        raise PreventUpdate
        
    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]
        
    if 'All' in selected_genes:
        selected_genes = ['All']
    
    if 'None' in selected_genes and len(selected_genes) > 1:
        selected_genes.remove('None')
    
    fig = gen_figure(selected_genes, ACTIVE_DATA['name'])
    
    end = datetime.now()
    print(f'returning from callback at {end} after {end-start}')
    
    return fig


@app.callback(
    Output('gene-div', 'children'),
    [Input('data-select', 'value')]
)
def select_data(folder):
    
    if folder is None:
        return None
    
    if not os.path.exists(config['local_store']):
        os.mkdir(config['local_store'])
        
    local_folder = os.path.join(config['local_store'], folder)
    
    def findlocal(name, local_folder=local_folder):
        return os.path.join(local_folder, name)
    
    # the desired folder doesn't exist, so we must fetch from s3
    if not os.path.exists(local_folder):
        download_s3_folder(folder, local_folder)
        
    assert os.path.exists(local_folder), f'Unable to fetch folder {folder} from s3.'
    
    print(f'{findlocal(config["mesh_name"])}, {findlocal(config["pcd_name"])}')
    
    # If we already have the mesh file for this, read from json.
    # else, generate it and save it.
    if os.path.exists(findlocal(config['mesh_name'])):
        mesh = mesh_from_json(findlocal(config['mesh_name']))
    else:
        mesh = gen_mesh(findlocal(config['img_name']),
                        outfile=findlocal(config['mesh_name']))
    
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
    
    return dcc.Dropdown(
        id='gene-select',
        options=[{'label': i, 'value': i} for i in genes],
        value='None',
        multi=True,
        placeholder='Select gene(s)',
        style={}
    )    
    
    
######## App layout and initialization ########

app.layout = html.Div(children=[
    html.H1(children='webfish test', style={'margin': 'auto'}),
    dbc.Alert("This is an info alert. Good to know!", color="info"),
    
    html.Div([
        dcc.Dropdown(
            id='data-select',
            options=[{'label': i, 'value': i} for i in possible_folders],
            placeholder='Select a data folder',
            style={}
        ),
    ], id='selector-div', style={'width': '200px', 'font-color': 'black'}),
    
    html.Div([dcc.Dropdown(id='gene-select')],
             id='gene-div', style={'width': '200px'}),
    
    html.Div([
        dcc.Graph(
            id='test-graph',
        )
    ]),
], style={'margin': 'auto', 'width': '800px'})

if __name__ == '__main__':
    app.run_server(debug=True)
