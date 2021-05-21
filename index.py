import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from lib.client import DataClient
from app import app, config, s3_client
from pages import datavis, dotdetection, splash
from pages._common import ComponentManager, dataset_form

data_client = DataClient(config=config, s3_client=s3_client)

# Sync with no pagename supplied, grabbing all datasets possible.
data_client.sync_with_s3()

# Save the dataset record to compare to individual pages
all_datasets = data_client.datasets.copy()

# Convenience dict with the page name (short), title (for display), and description
# TODO: Put this into DataClient
page_index = {k: {'title': v.get('title', k),
                  'description': v.get('description', '')
                 }
              for k, v in config['pages'].items()}

# The default "splash" or "home" page displayed on load
splash_tab = {'tab-splash':
                  dcc.Tab(id='tab-splash', label='Home', value='home')}

page_tabs = {}
page_tooltips = {}

for k, v in page_index.items():
    # Generate a Tab object for each page besides the splash,
    # disabled by default
    page_tabs[f'tab-{k}'] = dcc.Tab(
        id=f'tab-{k}',
        label=v['title'],
        value=k,
        disabled=True
    )

    # Add a Tooltip with the page description to each page tab that has a
    # description entry.
    if v['description']:
        page_tooltips[f'tooltip-{k}'] = dbc.Tooltip(
            v['description'],
            target=f'tab-{k}',
            placement='bottom'
        )

# Combine the splash tab and the page tabs dicts
all_tabs = splash_tab | page_tabs
tabs_and_tooltips = all_tabs | page_tooltips

# Our components are the global user, dataset selectors plus the tab machinery
# Set up a component group for all tabs and one with just the non-splash tabs
index_cm = ComponentManager(
    dataset_form | tabs_and_tooltips,
    component_groups={
        'all-tabs': list(all_tabs.keys()),
        'main-tabs': list(page_tabs.keys()),
        'page-tooltips': list(page_tooltips.keys())
    }
)


def select_page(pagename, user, dataset):
    """
    select_page
    -----------
    Switch to a page in the DataClient and determine if it has content for
    the given user, dataset combination.
    """
    data_client.pagename = pagename
    data_client.sync_with_s3(download=False)

    page_dataset = data_client.datasets.query(
        'user==@user and dataset==@dataset')

    return not page_dataset.empty


app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='wf-store', storage_type='session', data={}),

        html.H1('webfish app'),

        # Global user and dataset selectors
        html.Div([
            index_cm.component(
                'user-select',
                options=[{'label': u, 'value': u}
                         for u in sorted(all_datasets['user'].unique())]
            ),
            index_cm.component('dataset-select', disabled=True),
        ], style={'width': '500px'}),

        # Tabs container
        dcc.Tabs(
            id='all-tabs',
            children=index_cm.component_group('all-tabs', tolist=True),
            style={'width': '500px'}
        ),

        *index_cm.component_group('page-tooltips', tolist=True),

        # The main Div that will be updated with the selected tab's content
        html.Div(id='content-main', style={'width': '100%', 'height': '100%'})
    ], style={'margin': 'auto'}
)


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
    [Output(k, 'disabled') for k in page_tabs.keys()],
    Input('dataset-select', 'value'),
    State('user-select', 'value')
)
def select_dataset(dataset, user):
    """
    select_dataset
    --------------
    Updates the page tabs' disabled state according to whether each one
    is able to display content for the selected user and dataset.

    We only operate on `page_tabs` because the splash/home tab should
    always be active.
    """

    if not dataset or not user:
        return tuple([True] * len(page_tabs.keys()))

    return tuple([not select_page(p, user, dataset) for p in page_index.keys()])


@app.callback(
    Output('content-main', 'children'),
    Input('all-tabs', 'value'),
)
def tab_handler(tabval):
    """
    tab_handler
    -----------
    Actually handle the selected tab name by populating the main
    content Div with the appropriate layout variable, defined in
    each individual page file.

    If we get anything unrecognized, return the home page.
    """
    if tabval == 'datavis':
        return datavis.layout
    elif tabval == 'dotdetection':
        return dotdetection.layout
    else:
        return splash.layout


@app.callback(
    Output('url', 'pathname'),
    Output('all-tabs', 'value'),
    Input('url', 'pathname'),
    Input('all-tabs', 'value')
)
def sync_tab_url(pathname, tabval):
    """
    sync_tab_url
    ------------
    Two-way syncing of the relative path in the URL bar and the selected
    tab.
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    print(f'triggered by {trigger_id} with pathname {pathname} and tabval {tabval}')

    pathname = pathname.strip('/')

    tabs_and_splash = ['home'] + list(page_index.keys())

    if trigger_id == 'url' or tabval not in tabs_and_splash:
        newval = pathname
    else:
        newval = tabval

    if newval not in tabs_and_splash:
        return '', ''

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
