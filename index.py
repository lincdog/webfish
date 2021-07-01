import os
import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, config
from lib.util import empty_or_false
import importlib
from pages.common import (
    ComponentManager,
    get_all_datasets,
    sync_with_s3
)

# Set up the root logger
logger = logging.getLogger('webfish')
logger.setLevel(logging.DEBUG)
hand = logging.FileHandler('wflogger.log')
hand.setLevel(logging.DEBUG)

formatter = logging.Formatter(
        f'[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
hand.setFormatter(formatter)
logger.addHandler(hand)

logger.info('At the top of index.py')

# Convenience dict with:
# - key: shorthand page names, used as tab and URL values
# - title: for display on tabs, headers etc
# - description: longer information used in tooltips, etc
# - module: the imported module object corresponding to this page file, from
#   which we will get the layout for the page.

# TODO: Put this into DataClient?
page_index = {
    'home': {
        'title': 'Home',
        'description': '',
        'module': importlib.import_module('pages.splash')
    }
}

for k, v in config['pages'].items():
    try:
        modname = v.get('file', '').removesuffix('.py')
        fullname = f'pages.{modname}'

        page_index[k] = {
            'title': v.get('title', k),
            'description': v.get('description', ''),
            'module': importlib.import_module(fullname)
        }
    except (ImportError, ModuleNotFoundError) as e:
        print(f'Import error: {e}')

# The global user and dataset selectors, as well as the Sync with S3 button
dataset_form = {
    'user-select': dbc.Select(
        id='user-select',
        placeholder='Select a user',
        persistence=True,
        persistence_type='session'
    ),
    'dataset-select': dbc.Select(
        id='dataset-select',
        placeholder='Select a dataset',
        persistence=True,
        persistence_type='session'
    ),
    's3-sync-button':
        dbc.Button(
            'Sync data and analyses',
            id='s3-sync-button',
            n_clicks=0,
            color='primary'
        ),
    's3-sync-tooltip':
        dbc.Tooltip(
            'Update the app\'s manifest of analysis and raw image files. '
            'Automatically runs every 5 minutes as well.',
            target='s3-sync-button',
            placement='bottom',
        ),
    # Interval that fires every 5 minutes (units in milliseconds) to sync with s3
    's3-sync-interval':
        dcc.Interval(id='s3-sync-interval', interval=5 * 60 * 1000, n_intervals=0),
}


def get_users():
    return [{'label': u, 'value': u}
            for u in sorted(get_all_datasets()['user'].unique())]


page_tabs = {}
page_tooltips = {}

for k, v in page_index.items():
    # Generate a Tab object for each page besides the splash,
    # disabled by default
    page_tabs[f'tab-{k}'] = dbc.Tab(
        tab_id=k,
        id=f'tab-{k}',
        label=v['title'],
        disabled=False
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
tabs_and_tooltips = page_tabs | page_tooltips

# Our components are the global user, dataset selectors plus the tab machinery
index_cm = ComponentManager(
    dataset_form | tabs_and_tooltips,
    component_groups={
        'main-tabs': list(page_tabs.keys()),
        'page-tooltips': list(page_tooltips.keys())
    }
)

logger.info('Finished importing pages and setting up Index components')


@app.callback(
    Output('s3-sync-div', 'children'),
    Output('user-select', 'options'),
    Input('s3-sync-button', 'children'),
    State('s3-sync-button', 'n_clicks'),
    State('s3-sync-interval', 'n_intervals'),
)
def finish_client_s3_sync(contents, n_clicks, n_intervals):
    if n_clicks + n_intervals != 1:
        raise PreventUpdate

    if ' Syncing...' not in contents:
        raise PreventUpdate

    sync_with_s3()

    return ([index_cm.component('s3-sync-button'),
            index_cm.component('s3-sync-interval')],
            get_users())


@app.callback(
    Output('s3-sync-button', 'children'),
    Input('s3-sync-button', 'n_clicks'),
    Input('s3-sync-interval', 'n_intervals')
)
def init_client_s3_sync(n_clicks, n_intervals):
    if n_intervals == 1:
        print('interval fired')

    print(n_clicks, n_intervals)

    # Initialize sync when button is clicked OR interval fires, not both -
    # to prevent re-syncing if the button is clicked during a sync or the
    # interval fires during a sync, before the counts have reset.
    if n_clicks + n_intervals == 1:
        return [dbc.Spinner(size='sm'), ' Syncing...']
    else:
        raise PreventUpdate


@app.callback(
    Output('dataset-select', 'options'),
    Output('dataset-select', 'value'),
    Output('dataset-select', 'disabled'),
    Input('user-select', 'value')
)
def select_user(user):
    datasets = get_all_datasets().query('user==@user')['dataset'].unique()

    return (
        [{'label': d, 'value': d} for d in sorted(datasets)],
        None,
        len(datasets) == 0
    )


@app.callback(
    Output('content-main', 'children'),
    Input('all-tabs', 'active_tab'),
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
    entry = page_index.get(tabval, page_index['home'])

    if hasattr(entry['module'], 'layout'):
        return entry['module'].layout
    else:
        return page_index['home']['module'].layout


@app.callback(
    Output('url', 'pathname'),
    Output('all-tabs', 'active_tab'),
    Input('url', 'pathname'),
    Input('all-tabs', 'active_tab')
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
        return 'home', 'home'

    return newval, newval


app.layout = dbc.Container(
    fluid=True,
    children=[
        # Location to track the URL bar contents
        dcc.Location(id='url', refresh=False),
        # Store to store state locally (currently unused)
        dcc.Store(id='wf-store', storage_type='session', data={}),

        dbc.Row([
            dbc.Col([
                dbc.NavbarSimple(
                    [
                        dbc.NavItem(dbc.NavLink(
                            'View on GitHub',
                            href='https://github.com/CaiGroup/web-ui'
                        )),
                        dbc.NavItem(dbc.NavLink(
                            'Cai Lab home',
                            href='https://spatial.caltech.edu'
                        ))
                    ],
                    brand='Webfish',
                    brand_href='/home',
                    color='primary',
                    dark=True
                )
            ]),
        ]),

        dbc.Row([
            dbc.Col([
                # Global user and dataset selectors
                dbc.Card(dbc.CardBody([
                    index_cm.component('user-select', options=get_users()),
                    index_cm.component('dataset-select', disabled=True),
                    html.Div([
                        index_cm.component('s3-sync-button'),
                        index_cm.component('s3-sync-tooltip'),
                        index_cm.component('s3-sync-interval')
                    ], id='s3-sync-div'),
                ]), style={'margin-top': '10px'}),
            ], width=4),
            dbc.Col([
                # Tabs container
                dbc.Tabs(
                    id='all-tabs',
                    active_tab='home',
                    children=index_cm.component_group('main-tabs', tolist=True),
                ),

                *index_cm.component_group('page-tooltips', tolist=True),
            ], style={'margin-top': '20px'}, align='end'),

            html.Hr()
        ]),

        # This Row is populated by the page.layout of the selected tab.
        # Note that dbc.Row only works properly if its children are
        # dbc.Col objects. So each page.layout should be a list of these.
        dbc.Row(id='content-main',
                style={'margin': '10px'}),

    ], style={'margin': 'auto'}
)


if __name__ == '__main__':
    logger.info('Entering server loop')

    hostip = os.environ.get('WEBFISH_HOST', '127.0.0.1')
    hostport = os.environ.get('WEBFISH_PORT', '8050')
    app.run_server(
        debug=True,
        host=hostip,
        port=hostport,
        #dev_tools_props_check=False,
        #processes=4,
        #threaded=False
    )
