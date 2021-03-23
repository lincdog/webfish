import json
import dash
import dash_core_components as dcc
import dash_html_components as html
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

pz,py,px = dots_pcd[['z', 'y', 'x']].values.T

figdata = [
    go.Mesh3d(x=x, y=y, z=z,
              i=i, j=j, k=k,
        color='lightgray',
        opacity=0.7,
        hoverinfo='skip',
    ),
    go.Scatter3d(x=px, y=py, z=pz,
        mode='markers',
        marker=dict(
            size=1,
            color=dots_pcd['geneColor'],
            opacity=1,
            symbol='circle',
        ),
        hoverinfo='text',
        hovertext=dots_pcd['geneID']
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
#fig.show(renderer='jupyterlab')


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Tester', style={'font-family': 'sans-serif'}),
    dcc.Graph(
        id='test-graph',
        figure=fig
    )
], style={'align': 'center'})

if __name__ == '__main__':
    app.run_server(debug=True)
