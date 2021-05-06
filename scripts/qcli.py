from importlib import reload

import yaml
import sys
import lib.cloud as cloud


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

    config = yaml.load(open('../consts.yml'), Loader=yaml.Loader)
    s3c = cloud.S3Connect(config=config)
    server = cloud.DataServer(config=config, s3_client=s3c)

    return config, s3c, server
