from importlib import reload

import yaml
import logging
import sys
import lib.cloud as cloud

server_logger = logging.getLogger('lib.cloud.server')
client_logger = logging.getLogger('lib.cloud.client')


def qclient(pagename='datavis'):
    global cloud
    cloud = reload(cloud)

    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)
    s3c = cloud.S3Connect(config=config)
    client = cloud.DataClient(config=config, s3_client=s3c, pagename=pagename)

    return config, s3c, client


def qserver():
    global cloud
    cloud = reload(cloud)

    server_logger.setLevel(logging.DEBUG)

    rth = logging.StreamHandler(stream=sys.stdout)
    rth.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
    rth.setFormatter(formatter)

    server_logger.addHandler(rth)

    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)
    s3c = cloud.S3Connect(config=config)
    server = cloud.DataServer(config=config, s3_client=s3c)

    return config, s3c, server
