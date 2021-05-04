import os
import sys
import yaml
import time
import signal
import pandas as pd
from pathlib import Path, PurePath
from argparse import ArgumentParser

os.chdir('/home/lombelet/cron/webfish')
sys.path.extend([os.getcwd()])
from lib import cloud
from lib.util import ls_recursive

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
    config = yaml.load(open('../consts.yml'), Loader=yaml.Loader)
    try:
        old_patterns = open(Path(config['sync_folder'], 'source_patterns')).read()
    except FileNotFoundError:
        old_patterns = ''

    old_patterns = old_patterns.split()

    s3c = cloud.S3Connect(config=config)

    dm = cloud.DataServer(config=config, s3_client=s3c)

    return dm, old_patterns


def stat_compare(dm):
    try:
        listmtime = float(open(Path(dm.sync_folder, 'TIMESTAMP'), 'r').read().strip())

    except FileNotFoundError:
        listmtime = 0

    dirlist = ls_recursive(root=dm.master_root, level=dm.dataset_nest, flat=True)

    modified = []

    for d in dirlist:
        if os.stat(Path(dm.master_root, d)).st_mtime > listmtime:
            modified.append(Path(dm.master_root, d))

    return modified


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


# TODO: Before uploading file, allow hook for server-side generator class
#   that will process the files before uploading them - e.g. compression, etc
def datafile_search(dm, diff, dryrun, deep=False):
    if deep or dryrun:
        diff = None  # setting the folders arg of find_datafiles to none looks in ALL folders
    elif not diff:
        return pd.DataFrame()  # if we get an empty list, return an empty DF

    for page in dm.pagenames:
        _, new_files = dm.find_datafiles(page=page, folders=diff)

        pending_csv = Path(dm.sync_folder, f'{page}_pending.csv')

        if pending_csv.exists():
            pending_files = pd.read_csv(pending_csv)
        else:
            pending_files = pd.DataFrame()

        new_files = pd.concat([new_files, pending_files])
        new_files.drop_duplicates(subset='filename', inplace=True, ignore_index=True)

        if new_files.empty:
            continue

        i = 0

        remaining_files = new_files.copy()

        signal.signal(signal.SIGINT, sigint_write_pending)

        if not args.dryrun:
            for row in new_files.itertuples():
                if i % 50 == 0:
                    print(f'done with {i} files, {len(new_files) - i} to go')
                i += 1

                key_prefix = PurePath(dm.analysis_folder)
                keyname = key_prefix / Path(row.filename)

                try:
                    dm.client.client.upload_file(
                        str(Path(dm.master_root, row.filename)),
                        Bucket=dm.bucket_name,
                        Key=str(keyname)
                    )

                    remaining_files.drop(index=row.Index, inplace=True)

                except Exception as ex:
                    print(f'problem uploading file {row.filename}: {ex}')

            if not remaining_files.empty:
                remaining_files.to_csv(pending_csv, index=False)

    return new_files


if __name__ == '__main__':
    args = process_args()

    dm, old_patterns = init_server()

    if args.fresh:
        deep = True
        for f in Path(dm.sync_folder).iterdir():
            if not f.name == 'source_patterns':
                f.unlink()

    new_folders = stat_compare(dm)

    new_patterns = []
    for p in dm.pages.values():
        new_patterns.extend(list(p.source_patterns.values()))

    deep = sorted(old_patterns) != sorted(new_patterns)

    new_files = datafile_search(dm, new_folders, dryrun=args.dryrun, deep=deep)

    with open(Path(dm.sync_folder, 'TIMESTAMP'), 'w') as ts:
        ts.write(str(time.time()))
