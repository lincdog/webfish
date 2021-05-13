import dash
# from flask_caching import Cache
import dash_bootstrap_components as dbc
import yaml
from lib.cloud import S3Connect

####### Globals #######

config_file = 'consts.yml'
config = yaml.load(open(config_file), Loader=yaml.Loader)

# this will loop for 120 seconds, checking every second for the credentials file
# specified by the environment variable config['credentials'].
s3_client = S3Connect(config=config, wait_for_creds=True, wait_timeout=120)

############# Begin app code ############

THEME = getattr(dbc.themes, config.get('theme', 'MINTY').upper())
app = dash.Dash(
    __name__,
    external_stylesheets=[THEME],
    suppress_callback_exceptions=True,
    update_title=None,
)

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'filesystem',
#    'CACHE_DIR': 'cache-directory'
#})
