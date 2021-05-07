import os
import sys
WF_HOME = '/home/lombelet/cron/webfish_sandbox/webfish'
sys.path.append(WF_HOME)
import yaml
import time
import signal
import pandas as pd
from pathlib import Path, PurePath
from argparse import ArgumentParser
from lib import cloud
from lib.util import (
    ls_recursive,
    find_matching_files,
    empty_or_false,
    notempty
)


TIMESTAMP = 'TIMESTAMP'
PATTERNS = 'input_patterns'

# TODO: Much of this functionality would better be integrated into cloud.DataClient itself


def process_args():
    parser = ArgumentParser(description='HPC-side script to sync Cai Lab datasets to S3 storage')

    parser.add_argument('--dryrun', action='store_true',
                        help='Only update monitoring information, do not upload to s3.')

    parser.add_argument('--fresh', action='store_true',
                        help='Erase all stored monitoring files and regenerate them. '
                             'Note: if --dryrun is not specified, will begin '
                             'uploading ALL datafiles.')

    return parser.parse_args()


def init_server():
    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)
    try:
        old_patterns = open(Path(config['sync_folder'], PATTERNS)).read()
    except FileNotFoundError:
        old_patterns = ''

    old_patterns = old_patterns.split()

    s3c = cloud.S3Connect(config=config)

    dm = cloud.DataServer(config=config, s3_client=s3c)

    return dm, old_patterns


def stat_compare(dm):
    try:
        listmtime = float(open(
            Path(dm.sync_folder, TIMESTAMP), 'r').read().strip())
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

    if pending_csv and df:
        try:
            df.to_csv(pending_csv, index=False)
            print('Keyboard interrupt received, saving pending uploads and exiting.')
            sys.exit(0)
        except IOError:
            print('Keyboard interrupt received, failed to save pending uploads.')
            sys.exit(1)
    else:
        print('Keyboard interrupt received, unable to locate pending uploads')
        sys.exit(1)


def datafile_search(dm, diffs, mtime, dryrun=False, deep=False):

    if deep or dryrun:
        # setting the folders arg of find_datafiles to none looks in ALL folders
        diffs = (None, None)

    for page in dm.pagenames:

        pending_csv = Path(dm.sync_folder, f'{page}_pending.csv')
        pending_files = pd.DataFrame()
        if pending_csv.exists():
            pending_files = pd.read_csv(pending_csv)

        if diffs == ([], []) and pending_files.empty:
            continue

        new_files, new_datasets = dm.find_page_files(
            page=page,
            source_folders=diffs[0],
            raw_folders=diffs[1],
            since=mtime,
            sync=True
        )

        new_files = pd.concat([new_files, pending_files])
        new_files.drop_duplicates(subset='filename', inplace=True, ignore_index=True)

        if new_files.empty:
            continue

        signal.signal(signal.SIGINT, sigint_write_pending)

        if not args.dryrun:
            remaining_files = dm.upload_to_s3(page, new_files, progress=100)

            if not remaining_files.empty:
                remaining_files.to_csv(pending_csv, index=False)
        else:
            new_files.to_csv(Path(dm.sync_folder, f'{page}_dryrun.csv'))

    return new_files


def main(args):
    dm, old_patterns = init_server()

    breakpoint()

    if args.fresh:
        deep = True
        for f in Path(dm.sync_folder).iterdir():
            if not f.name == PATTERNS:
                f.unlink()

    source_diffs, raw_diffs, mtime = stat_compare(dm)

    new_patterns = []
    for p in dm.pages.values():
        new_patterns.extend(list(p.input_patterns.values()))

    deep = sorted(old_patterns) != sorted(new_patterns)

    new_files = datafile_search(
        dm,
        (source_diffs, raw_diffs),
        mtime,
        dryrun=args.dryrun,
        deep=deep
    )

    now = time.time()

    if args.dryrun:
        verb = 'Found'
    else:
        verb = 'Uploaded'

    print(f'{verb} {len(new_files)} files, finishing at {now}')

    with open(Path(dm.sync_folder, TIMESTAMP), 'w') as ts:
        ts.write(str(now))


if __name__ == '__main__':
    os.chdir(WF_HOME)
    args = process_args()
    main(args)
