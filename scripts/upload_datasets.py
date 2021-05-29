import os
import sys
import yaml
import time
import logging
import atexit
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePath
from argparse import ArgumentParser
WF_HOME = os.environ.get('WF_HOME', '/home/lombelet/cron/webfish_sandbox/webfish')
sys.path.append(WF_HOME)
from lib.server import DataServer
from lib.core import S3Connect


def process_args():
    parser = ArgumentParser(description='HPC-side script to sync Cai Lab datasets to S3 storage')

    parser.add_argument('--dryrun', action='store_true',
                        help='Only update monitoring information, do not upload to s3.')

    parser.add_argument('--fresh', action='store_true',
                        help='Erase all stored monitoring files and regenerate them. '
                             'Note: if --dryrun or --use-s3-only is not specified, will begin '
                             'uploading ALL datafiles.')

    parser.add_argument('--check-s3', action='store_true',
                        help='List actual keys from the s3 bucket to accurately determine '
                             'the difference between local files and cloud storage.'
                             ' Somewhat slow to list many objects.')

    parser.add_argument('--use-s3-only', action='store_true',
                        help='When deciding what files need to be uploaded, '
                             'ONLY consider those that are present locally '
                             'but missing from s3 - ignore modification times,'
                             ' new input patterns, and any current pending files')

    return parser.parse_args()


def init_server():
    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)

    s3c = S3Connect(config=config)

    dm = DataServer(config=config, s3_client=s3c)
    dm.read_local_sync()

    return dm


def search_and_upload(dm, mtime, use_s3_only=False, check_s3=False, dryrun=False):

    results = {}

    tmp, _ = dm.find_files()
    results['all_files'] = len(tmp)
    del tmp

    # read in the s3 keys, unless --check-s3 is specified, from the local
    # cached listing.
    dm.check_s3_contents(use_local=(not check_s3))

    pending, uploaded = dm.upload_to_s3(
        since=mtime,
        do_pending=True,
        run_preuploads=True,
        do_s3_diff=True,
        use_s3_only=use_s3_only,
        progress=100,
        dryrun=dryrun
    )

    results.update(dict(pending_count=len(pending),
                        uploaded_count=uploaded))

    return results


def atexit_release(file):
    file.unlink(missing_ok=True)


def main(args):

    dm = init_server()

    mtime = dm.local_sync['timestamp'] or 0

    lock = Path(dm.sync_folder, LOCKFILE)

    if lock.exists():
        logger.info('Lockfile exists, exiting.')
        return 0

    with open(lock, 'w') as lockfp:
        lockfp.write('{0} {1}'.format(os.getpid(), time.time()))

    atexit.register(atexit_release, lock)

    if args.fresh:
        mtime = 0
        # Delete all monitoring files
        for f in Path(dm.sync_folder).iterdir():
            if f.name != LOCKFILE:
                f.unlink()

    results = search_and_upload(
        dm,
        mtime,
        use_s3_only=args.use_s3_only,
        check_s3=args.check_s3,
        dryrun=args.dryrun,
    )

    logger.info(f'Results: {results}')

    dm.save_and_sync(
        timestamp=True,
        patterns=True,
        upload=(not args.dryrun)
    )

    lock.unlink(missing_ok=True)


if __name__ == '__main__':
    os.chdir(WF_HOME)
    LOCKFILE = f'upload_datasets.lck'

    logger = logging.getLogger('lib.server')
    logger.setLevel(logging.DEBUG)

    rth = RotatingFileHandler('upload_datasets.log', maxBytes=2 ** 16, backupCount=4)
    rth.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        f'<pid {os.getpid()}>[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
    rth.setFormatter(formatter)

    logger.addHandler(rth)

    args = process_args()
    logger.info(f'upload_datasets: starting with arguments: {args}')
    main(args)
