import os
import sys
import yaml
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
WF_HOME = os.environ.get('WF_HOME', '/home/lombelet/cron/webfish_sandbox/webfish')
sys.path.append(WF_HOME)
os.chdir(WF_HOME)
from lib.server import DataServer
from lib.core import S3Connect


def init_server():
    """
    init_server
    -----------
    Initialize a lib.server.DataServer instance and an S3 client. Use read_local_sync
    to gather current dataset and file information, including the previous TIMESTAMP
    and input patterns list that allow us to determine what files to upload.
    """
    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)

    s3c = S3Connect(config=config)

    dm = DataServer(config=config, s3_client=s3c)
    dm.read_local_sync()

    return dm


if __name__ == '__main__':
    logger = logging.getLogger('lib.server')
    logger.setLevel(logging.DEBUG)

    rth = RotatingFileHandler('upload_logging.log', maxBytes=2**18, backupCount=1)
    rth.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        f'<pid {os.getpid()}>[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
    rth.setFormatter(formatter)

    logger.addHandler(rth)

    dm = init_server()

    if not dm.analysis_log:
        logger.warning('No log filename defined in config')
        sys.exit(1)

    analysis_log_local = Path(dm.file_locations['source']['root'],
                              dm.analysis_log)

    analysis_log_key = Path(dm.sync_folder, dm.analysis_log)

    if Path(analysis_log_local).is_file():
        logger.info(f'Attempting to upload log file at {analysis_log_local} '
                       f'to key {analysis_log_key}')
        try:
            dm.client.client.upload_file(
                str(analysis_log_local),
                Bucket=dm.bucket_name,
                Key=str(analysis_log_key)
            )
        except Exception as e:
            logger.error(f'Error uploading log file: {e}')
    else:
        logger.error(f'Log file {analysis_log_local} does not exist')
        sys.exit(1)

    logger.info('Successfully upload logfile, exiting')
    sys.exit(0)