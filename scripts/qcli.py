from importlib import reload

import yaml
import logging
import sys
import lib.core
import lib.client
import lib.server

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
sh.setFormatter(formatter)

server_logger = logging.getLogger('lib.server')
client_logger = logging.getLogger('lib.client')

server_logger.setLevel(logging.DEBUG)
server_logger.addHandler(sh)


def qclient():
    lib.core = reload(lib.core)
    lib.client = reload(lib.client)

    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)
    s3c = lib.core.S3Connect(config=config)
    client = lib.client.DataClient(config=config, s3_client=s3c)

    return config, s3c, client


def qserver(test_mode=False):
    lib.core = reload(lib.core)
    lib.server = reload(lib.server)

    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)
    s3c = lib.core.S3Connect(config=config)
    server = lib.server.DataServer(config=config, s3_client=s3c, test_mode=test_mode)

    return config, s3c, server
