import json
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



MESH_FILE = 'data/labeled_16x_mesh.json'
PCD_FILE = 'data/dots_um_with_colors.csv'

with open(MESH_FILE, 'r') as mf:
    cell_mesh = json.load(mf)

with open(PCD_FILE, 'r') as pf:
    dots_pcd = pd.read_csv(pf)

cell_mesh['verts'] = np.array(cell_mesh['verts'])
cell_mesh['faces'] = np.array(cell_mesh['faces'])

z,y,x = cell_mesh['verts'].T
i,j,k = cell_mesh['faces'].T

unique_genes, gene_counts = np.unique(dots_pcd['geneID'], return_counts=True)

possible_genes = ['None', 'All'] + list(np.flip(unique_genes[np.argsort(gene_counts)]))


############# Begin app code
THEME = dbc.themes.MINTY
app = dash.Dash(__name__, external_stylesheets=[THEME])

print(THEME)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

@cache.memoize()
def query_df(selected_genes):
    if 'All' in selected_genes:
        dots_filt = dots_pcd
    elif selected_genes == ['None']:
        dots_filt = dots_pcd.query('geneID == "NOT__A__GENE"')
    else:
        dots_filt = dots_pcd.query('geneID in @selected_genes')
    
    return dots_filt.to_json()

def dataframe(selected_genes):
    return pd.read_json(query_df(selected_genes))

@cache.memoize()
def gen_figure(selected_genes):
    
    dots_filt = dataframe(selected_genes)
    
    pz,py,px = dots_filt[['z', 'y', 'x']].values.T
    
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
                    color=dots_filt['geneColor'],
                    opacity=1,
                    symbol='circle',
                ),
                hoverinfo='text',
                hovertext=dots_filt['geneID']
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


@app.callback(
    Output('test-graph', 'figure'),
    Input('gene-select', 'value'),
    #State('test-graph', 'figure')
    )
def update_figure(selected_genes):
    
    start = datetime.now()
    print(f'starting callback at {start}')
    
    if len(selected_genes) == 0:
        raise PreventUpdate
        
    if not isinstance(selected_genes, list):
        selected_genes = [selected_genes]
        
    if 'All' in selected_genes:
        selected_genes = ['All']
    
    if 'None' in selected_genes and len(selected_genes) > 1:
        selected_genes.remove('None')
    
    fig = gen_figure(selected_genes)
    
    end = datetime.now()
    print(f'returning from callback at {end} after {end-start}')
    
    return fig


app.layout = html.Div(children=[
    html.H1(children='webfish test', style={'margin': 'auto'}),
    dbc.Alert("This is an info alert. Good to know!", color="info"),
    
    html.Div([
            dcc.Dropdown(
                id='gene-select',
                options=[{'label': i, 'value': i} for i in possible_genes],
                value='None',
                multi=True,
                placeholder='Select gene(s)',
                style={}
            ),
    ], style={'width': '200px', 'font-color': 'black'}),  
    html.Div([
        dcc.Graph(
            id='test-graph',
        )
    ]),
], style={'margin': 'auto', 'width': '800px'})

if __name__ == '__main__':
    app.run_server(debug=True)
