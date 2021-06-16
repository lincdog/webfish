import dash
import dash_bootstrap_components as dbc
import yaml
import logging

from lib.core import S3Connect

# Globals

config_file = 'consts.yml'
config = yaml.load(open(config_file), Loader=yaml.Loader)

# this will loop for 120 seconds, checking every second for the credentials file
# specified by the environment variable config['credentials'].
s3_client = S3Connect(config=config, wait_for_creds=True, wait_timeout=120)

# Begin app code

THEME = getattr(dbc.themes, config.get('theme', 'MINTY').upper())
app = dash.Dash(
    __name__,
    title='Webfish',
    external_stylesheets=[THEME],
    suppress_callback_exceptions=True,
    #update_title=None,
)

