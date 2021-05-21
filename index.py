import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from lib.client import DataClient
from app import app, config, s3_client
from pages import datavis, dotdetection
from pages._common import ComponentManager, dataset_form

data_client = DataClient(config=config, s3_client=s3_client)
data_client.sync_with_s3()

all_datasets = data_client.datasets.copy()

page_index = {
    k:
        {
            'title': v.get('title', k),
            'description': v.get('description', '')
        }
    for k, v in config['pages'].items()
}

main_tabs = {'tab-home': dcc.Tab(id='tab-home', label='Home page', value='tab-home')}

main_tabs |= {
    f'tab-{k}': dcc.Tab(id=f'tab-{k}', label=v['title'], value=k, disabled=True)
    for k, v in page_index.items()
}

index_cm = ComponentManager(
    dataset_form | main_tabs,
    component_groups={'main-tabs': list(main_tabs.keys())}
)


def select_page(pagename, user, dataset):
    data_client.pagename = pagename
    data_client.sync_with_s3(download=False)

    page_dataset = data_client.datasets.query(
        'user==@user and dataset==@dataset')

    return not page_dataset.empty


app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='wf-store', storage_type='session', data={}),
    html.H1('webfish app'),

    index_cm.component(
        'user-select',
        options=[{'label': u, 'value': u} for u in sorted(all_datasets['user'].unique())]
    ),
    index_cm.component('dataset-select', disabled=True),

    dcc.Tabs(
        id='main-tabs',
        children=index_cm.component_group('main-tabs', tolist=True),
        style={'width': '500px'}
    ),

    html.Div(id='content-main', style={'width': '100%', 'height': '100%'})
], style={'margin': 'auto'})


@app.callback(
    Output('dataset-select', 'options'),
    Output('dataset-select', 'value'),
    Output('dataset-select', 'disabled'),
    Input('user-select', 'value')
)
def select_user(user):
    datasets = all_datasets.query('user==@user')['dataset'].unique()

    return [{'label': d, 'value': d} for d in sorted(datasets)], None, False


@app.callback(
    [Output(f'tab-{p}', 'disabled') for p in valid_pagenames],
    Input('dataset-select', 'value'),
    State('user-select', 'value')
)
def select_dataset(dataset, user):
    if not dataset or not user:
        return tuple([True] * len(valid_pagenames))

    return tuple([not select_page(p, user, dataset) for p in valid_pagenames])


@app.callback(
    Output('content-main', 'children'),
    Input('main-tabs', 'value'),
)
def tab_handler(tabval):
    if tabval == 'datavis':
        return datavis.layout
    elif tabval == 'dotdetection':
        return dotdetection.layout
    else:
        return html.H1('404!!!!')


@app.callback(
    Output('url', 'pathname'),
    Output('main-tabs', 'value'),
    Input('url', 'pathname'),
    Input('main-tabs', 'value')
)
def sync_tab_url(pathname, tabval):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    pathname = pathname.strip('/')

    if trigger_id == 'url' or tabval not in valid_pagenames:
        newval = pathname
    else:
        newval = tabval

    if newval == '':
        newval = 'datavis'  # default page

    if newval not in valid_pagenames:
        raise PreventUpdate

    return newval, newval


if __name__ == '__main__':
    hostip = os.environ.get('WEBFISH_HOST', '127.0.0.1')
    hostport = os.environ.get('WEBFISH_PORT', '8050')
    app.run_server(
        debug=True,
        host=hostip,
        port=hostport,
        #dev_tools_props_check=False
    )
