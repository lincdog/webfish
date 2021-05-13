import os
import sys
import yaml
import time
import signal
import pandas as pd
from pathlib import Path, PurePath
from argparse import ArgumentParser
WF_HOME = os.get('WF_HOME', '/home/lombelet/cron/webfish_sandbox/webfish')
sys.path.append(WF_HOME)
from lib import cloud
from lib.util import find_matching_files

LOCKFILE = f'upload_datasets.lck'


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

    s3c = cloud.S3Connect(config=config)

    dm = cloud.DataServer(config=config, s3_client=s3c)
    dm.read_local_sync()

    return dm


def stat_compare(dm):
    try:
        listmtime = float(open(
            Path(dm.sync_folder, dm.sync_contents['timestamp']), 'r').read().strip())
    except FileNotFoundError:
        listmtime = 0

    source_dirlist, _ = find_matching_files(dm.master_root, dm.dataset_root)
    raw_dirlist, _ = find_matching_files(dm.raw_master_root, dm.raw_dataset_root)

    source_modified = []
    raw_modified = []

    for d in source_dirlist:
        if os.stat(Path(dm.master_root, d)).st_mtime > listmtime:
            source_modified.append(Path(dm.master_root, d))

    for d in raw_dirlist:
        if os.stat(Path(dm.raw_master_root, d)).st_mtime > listmtime:
            raw_modified.append(Path(dm.raw_master_root, d))

    return source_modified, raw_modified, listmtime


def dataset_compare(dm):
    old_datasets = pd.read_csv(Path(dm.sync_folder, 'all_datasets.csv'), dtype=str)

    new_datasets = dm.get_datasets()

    diff = set(old_datasets['folder']) ^ set(new_datasets['folder'])

    return list(diff)


def sigint_write_pending(signo, frame):
    pending_csv = frame.f_globals.get('pending_csv', None)
    df = frame.f_globals.get('remaining_files', None)

    lockfile = frame.f_globals.get('lock', None)

    if pending_csv and df:
        try:
            df.to_csv(pending_csv, index=False)
            print('Keyboard interrupt received, saving pending uploads and exiting.')
            sys.exit(0)
        except IOError:
            print('Keyboard interrupt received, failed to save pending uploads.')
            sys.exit(1)

    if lockfile:
        try:
            lockfile.unlink(missing_ok=True)
            sys.exit(0)
        except AttributeError:
            print('Keyboard interrupt received but failed to unlink lockfile.')
            sys.exit(1)

    sys.exit(1)


def search_and_upload(dm, mtime, use_s3_only=False, check_s3=False, dryrun=False):

    new_files = {}

    for pagename in dm.pagenames:
        tmp, _ = dm.find_page_files(
            pagename=pagename,
        )
        new_files[pagename] = len(tmp)
        del tmp

    # read in the s3 keys, unless --check-s3 is specified, from the local
    # cached listing.
    dm.check_s3_contents(use_local=(not check_s3))

    signal.signal(signal.SIGINT, sigint_write_pending)

    for pagename in dm.pagenames:
        dm.upload_to_s3(
            pagename,
            since=mtime,
            do_pending=True,
            run_preuploads=True,
            do_s3_diff=True,
            use_s3_only=use_s3_only,
            progress=100,
            dryrun=dryrun
        )

    return new_files


def main(args):

    dm = init_server()

    mtime = dm.local_sync['timestamp'] or 0

    lock = Path(dm.sync_folder, LOCKFILE)

    if lock.exists():
        return 0

    with open(lock, 'w') as lockfp:
        lockfp.write('{0} {1}'.format(os.getpid(), time.time()))

    if args.fresh:
        mtime = 0
        # Delete all monitoring files
        for f in Path(dm.sync_folder).iterdir():
            if f.name != LOCKFILE:
                f.unlink()

    new_files = search_and_upload(
        dm,
        mtime,
        use_s3_only=args.use_s3_only,
        check_s3=args.check_s3,
        dryrun=args.dryrun,
    )

    if args.dryrun:
        verb = 'Found'
    else:
        verb = 'Uploaded'

    print(f'{verb} {new_files} files')

    dm.save_and_sync(
        pagenames=None,
        timestamp=True,
        patterns=True,
        upload=(not args.dryrun)
    )

    lock.unlink(missing_ok=True)


if __name__ == '__main__':
    os.chdir(WF_HOME)
    args = process_args()
    breakpoint()
    main(args)
