import dash

# from flask_caching import Cache
import dash_bootstrap_components as dbc

import yaml

from cloud import S3Connect

####### Globals #######

config_file = 'consts.yml'
config = yaml.load(open(config_file), Loader=yaml.Loader)

s3_client = S3Connect(config=config)

############# Begin app code ############

THEME = getattr(dbc.themes, config.get('theme', 'MINTY').upper())
app = dash.Dash(
    __name__,
    external_stylesheets=[THEME],
    suppress_callback_exceptions=True
)

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'filesystem',
#    'CACHE_DIR': 'cache-directory'
#})
