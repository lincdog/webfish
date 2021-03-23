import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
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
app = dash.Dash(__name__)

figdata = [
    go.Mesh3d(x=x, y=y, z=z,
              i=i, j=j, k=k,
        color='lightgray',
        opacity=0.7,
        hoverinfo='skip',
    ),
    go.Scatter3d(
        name='dots',
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


@app.callback(
    Output('test-graph', 'figure'),
    Input('gene-select', 'value'),
    State('test-graph', 'figure')
    )
def update_figure(selected_genes, fig):
    
    if len(selected_genes) == 0:
        raise PreventUpdate
    
    fig = go.Figure(fig)
    
    if 'All' in selected_genes:
        dots_filt = dots_pcd
    elif selected_genes == ['None']:
        dots_filt = dots_pcd.query('geneID == "NOT__A__GENE"')
    else:
        dots_filt = dots_pcd.query('geneID in @selected_genes')

    pz,py,px = dots_filt[['z', 'y', 'x']].values.T

    newscatter_props = dict(
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
    

    newfig = fig.update_traces(newscatter_props, {'name': 'dots'})
    #fig.show(renderer='jupyterlab')

    return newfig


app.layout = html.Div(children=[
    html.H1(children='webfish test', style={'font-family': 'sans-serif', 'margin': 'auto'}),
    
    html.Div([
            dcc.Dropdown(
                id='gene-select',
                options=[{'label': i, 'value': i} for i in possible_genes],
                value='',
                multi=True
            ),
    ], style=dict(width='200px')),  
    html.Div([
        dcc.Graph(
            id='test-graph',
            figure=fig
        )
    ]),
], style={'margin': 'auto'})

if __name__ == '__main__':
    app.run_server(debug=True)
