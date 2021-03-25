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

from util import gen_mesh, gen_pcd_df

####### Globals #######

ACTIVE_DATA = {'name': None, 'mesh': None, 'dots': None}

CONSTS_FILE = 'consts.yml'
CONSTS = yaml.load(open(CONSTS_FILE), Loader=yaml.Loader)

for k, v in CONSTS.items():
    if k.upper() not in globals().keys():
        globals()[k.upper()] = v

#LOCAL_STORE = CONSTS['local_store']

#IMG_NAME = CONSTS['img_name']
#CSV_NAME = CONSTS['csv_name']

#MESH_NAME = CONSTS['mesh_name']
#PCD_NAME = CONSTS['pcd_name']

###### AWS Code #######
# assumes credentials & configuration are handled outside python in .aws directory or environment variables
s3 = boto3.resource('s3') 

BUCKET_NAME = 'lincoln-testing'
my_bucket = s3.Bucket(BUCKET_NAME)

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

##### Helper functions ######

def mesh_from_json(jsonfile):
    """
    mesh_from_json:
    take a json filename, read it in,
    and convert the verts and faces keys into numpy arrays.
    
    returns: dict of numpy arrays
    """
    cell_mesh = json.load(open(jsonfile))
    
    assert 'verts' in cell_mesh.keys(), f'Key "verts" not found in file {jsonfile}'
    assert 'faces' in cell_mesh.keys(), f'Key "faces" not found in file {jsonfile}'
    
    cell_mesh['verts'] = np.array(cell_mesh['verts'])
    cell_mesh['faces'] = np.array(cell_mesh['faces'])
    
    return cell_mesh

def populate_mesh(cell_mesh):
    """
    populate_mesh:
    take a mesh dictionary (like returned from `mesh_from_json`) and return the
    six components used to specify a plotly.graph_objects.Mesh3D
    
    returns: 6-tuple of numpy arrays: x, y, z are vertex coords; 
    i, j, k are vertex indices that form triangles in the mesh.
    """
    
    if cell_mesh is None:
        return None, None, None, None, None, None

    z,y,x = cell_mesh['verts'].T
    i,j,k = cell_mesh['faces'].T
    
    return x, y, z, i, j, k

def populate_genes(dots_pcd):
    """
    populate_genes:
    takes a dots dataframe and computes the unique genes present,
    sorting by most frequent to least frequent.
    
    returns: list of genes (+ None and All options) sorted by frequency descending
    """
    unique_genes, gene_counts = np.unique(dots_pcd['geneID'], return_counts=True)

    possible_genes = ['None', 'All'] + list(np.flip(unique_genes[np.argsort(gene_counts)]))
    
    return possible_genes
        

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
    # If dots is populated, grab it.
    # Otherwise, set the coords to None to create an empty Scatter3d.
    if ACTIVE_DATA['dots'] is not None:
        dots_filt = dataframe(selected_genes)
        pz,py,px = dots_filt[['z', 'y', 'x']].values.T
        color = dots_filt['geneColor']
        hovertext = dots_filt['geneID']
    else:
        pz,py,px = None, None, None
        color = None
        hovertext = None
    
    # If the mesh is present, populate it.
    # Else, create an empty Mesh3d.
    if ACTIVE_DATA['mesh'] is not None:
        x, y, z, i, j, k = populate_mesh(ACTIVE_DATA['mesh'])
    else:
        x, y, z, i, j, k = None, None, None, None, None, None
    
    figdata = [
        go.Mesh3d(x=x, y=y, z=z,
                  i=i, j=j, k=k,
            color='lightgray',
            opacity=0.7,
            hoverinfo='skip',
            ),
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
    ]

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
    Output('gene-select', 'options'),
    Input('data-select', 'value')
)
def select_data(folder):
    
    if folder is None:
        return None
    
    if not os.path.exists(LOCAL_STORE):
        os.mkdir(LOCAL_STORE)
        
    local_folder = os.path.join(LOCAL_STORE, folder)
    
    def findlocal(name, local_folder=local_folder):
        return os.path.join(local_folder, name)
    
    # the desired folder doesn't exist, so we must fetch from s3
    if not os.path.exists(local_folder):
        download_s3_folder(folder, LOCAL_STORE)
        
    assert os.path.exists(local_folder), f'Unable to fetch folder {folder} from s3.'
    
    # If we already have the mesh file for this, read from json.
    # else, generate it and save it.
    if os.path.exists(findlocal(MESH_NAME)):
        mesh = mesh_from_json(findlocal(MESH_NAME))
    else:
        mesh = gen_mesh(findlocal(IMG_NAME))
    
    # if we already have the processed point cloud DF for this, read it in.
    # else, generate it and save it.
    if os.path.exists(findlocal(PCD_NAME)):
        pcd = pd.read_csv(findlocal(PCD_NAME))
    else:
        pcd = gen_pcd_df(findlocal(CSV_NAME))
    
    ### Set global dots DF and mesh variables
    ACTIVE_DATA['name'] = folder
    ACTIVE_DATA['mesh'] = mesh
    ACTIVE_DATA['dots'] = pcd
    
    
    return populate_genes(pcd)
    
    
    
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
        dcc.Dropdown(
            id='gene-select',
            #options=[{'label': i, 'value': i} for i in possible_genes],
            value='None',
            multi=True,
            placeholder='Select gene(s)',
            style={}
        ),
    ], id='selector-div', style={'width': '200px', 'font-color': 'black'}),  
    html.Div([
        dcc.Graph(
            id='test-graph',
        )
    ]),
], style={'margin': 'auto', 'width': '800px'})

if __name__ == '__main__':
    app.run_server(debug=True)
