import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import time
import logging
import atexit
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePath
from argparse import ArgumentParser
WF_HOME = os.environ.get('WF_HOME', '/home/lombelet/cron/webfish_sandbox/webfish')

HISTORY_DIR = os.environ.get('WF_HISTORY_DIR', 'json_analyses')
HISTORY_FILE = os.environ.get('WF_ANALYSIS_HISTORY',
                              'wf_json_analyses_history.csv')
S3_PREFIX = os.environ.get('WF_S3_ANALYSIS_PREFIX', 'json_analyses/')
MASTER_ANALYSES = os.environ.get('WF_JSON_ANALYSES', '/groups/CaiLab/json_analyses')

LOCKFILE = 'download_json_analyses.lck'

sys.path.append(WF_HOME)
from lib.server import DataServer
from lib.core import S3Connect
from lib.util import sanitize

# * Read in request history file
# * Download contents of json_analyses/ prefix
# * Clear json_analyses/ prefix
# * Validate new json file(s)
#   * Keep them in a temporary local folder
#   * No identical analysis exists already in history file or other submissions
#   * Requested dataset actually exists
#   * Requested positions actually exist
#   * Fields are well-formed
# * Copy to json_analyses central folder
# * Return future location of files
# * Optional/future: Report on progress or maybe accept email to notify user
#   on starting/failure/completion

logger = logging.getLogger('download_json_analyses')


def init_server():
    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)

    s3c = S3Connect(config=config)

    dm = DataServer(config=config, s3_client=s3c)
    dm.read_local_sync(replace=True)

    return dm


def read_history_file(fname):
    history_df = pd.DataFrame()

    if Path(fname).is_file():
        try:
            history_df = pd.read_csv(fname, dtype=str)
        except Exception as e:
            logger.warning(f'Error opening json analyses history file '
                           f'{fname}: {e}')

            history_df = pd.DataFrame()
    else:
        logger.info(f'json analyses history file {fname} '
                    f'does not exist; will create.')

    return history_df


def download_s3_analysis_requests(dm):

    s3_files, _ = dm.client.list_objects(
        bucket_name=dm.bucket_name,
        prefix=S3_PREFIX)

    logger.info(f'Listed {len(s3_files)} objects in '
                f'{dm.bucket_name}/{S3_PREFIX}')

    s3_files_filt = [f for f in s3_files if f.suffix == '.json']

    errors = []
    successes = []

    for f in s3_files_filt:
        error = dm.client.download_s3_objects(
            dm.bucket_name,
            f.name,
            local_dir=HISTORY_DIR,
            prefix=S3_PREFIX
        )

        dm.client.client.delete_object(Bucket=dm.bucket_name, Key=f)

        if len(error) > 0:
            logger.error(f'Error downloading key {f}: {error}')
            errors.append(f)
        else:
            successes.append(f)

    return successes, errors


def convert_channel_entry(ent):
    if isinstance(ent, str):
        if ent == 'across':
            return ent
        else:
            return {
                'individual': ent.split('|')
            }
    elif isinstance(ent, dict):
        channels = ent['individual']

        return '|'.join(channels)

    raise TypeError(f'Bad decoding specification {ent}')


def validate_json(fname, history_df, dm):
    with open(fname) as f:
        new_json = json.load(f, parse_int=str, parse_float=str)

    existing_datasets = pd.concat([
        dm.all_datasets,
        dm.all_raw_datasets
    ], ignore_index=True)

    possible_users = existing_datasets['user'].unique()

    new_personal = new_json['personal']
    new_dataset = new_json['experiment_name']
    new_analysis = Path(fname).stem

    assert new_personal in possible_users, f'Unknown user {new_personal}'

    possible_datasets = existing_datasets.query(
        'user==@new_personal')['dataset'].unique()

    assert new_dataset in possible_datasets, f'Unknown raw dataset {new_dataset}'

    possible_analyses = existing_datasets.query(
        'user==@new_personal and dataset==@new_dataset')['analysis'].unique()

    assert new_analysis not in possible_analyses, \
        f'Analysis {new_analysis} already exists for this dataset and user.'

    relevant_history = history_df.query(
        'personal==@new_personal and experiment_name==@new_dataset')

    assert new_analysis not in relevant_history['analysis_name'], \
        f'Analysis {new_analysis} already exists for this dataset and user.'

    encoded_channels = convert_channel_entry(new_json['decoding'])
    new_json['decoding'] = encoded_channels
    new_json['analysis_name'] = new_analysis

    new_history = pd.concat([history_df,
                             pd.DataFrame([new_json])])

    columns_considered = new_history.columns.difference([
        'personal', 'experiment_name', 'analysis_name'])

    duplicates = new_history.duplicated(subset=columns_considered)
    if any(duplicates):
        dups = new_history.loc[duplicates, 'analysis_name'].values
        raise ValueError(f'Analyses {dups} uses the same parameters '
                         f'as request {new_analysis}.')

    return new_history.reset_index(drop=True)


def main(max_copy=10):
    dm = init_server()

    history_file = Path(HISTORY_DIR, HISTORY_FILE)
    history_df = read_history_file(history_file)

    new_success, new_fail = download_s3_analysis_requests(dm)

    logger.info(f'Downloaded {len(new_success)} new JSON files')

    if len(new_fail) > 0:
        logger.warning(f'Failed to download {len(new_fail)} new JSON files')

    to_copy = []

    for new_file in new_success:
        try:
            history_df = validate_json(new_file, history_df, dm)
            to_copy.append(new_file)
        except (AssertionError, ValueError) as e:
            logger.error(f'JSON file {new_file} was invalid for '
                         f'the following reason: ', e)

    if len(to_copy) > max_copy:
        logger.error(f'Too many ({len(to_copy)}) new analysis '
                     f'files (max set to {max_copy}) - ignoring '
                     f'these files: ', to_copy[max_copy:])

        history_df.drop(
            history_df.loc[np.isin(
                history_df['analysis_name'],
                [t.stem for t in to_copy[max_copy:]]
            )].index, inplace=True)

    successful_copies = []

    for file in to_copy[:max_copy]:
        try:
            file.rename(Path(MASTER_ANALYSES, file.name))
            successful_copies.append(file)
        except (PermissionError, OSError) as e:
            logger.error(f'Unable to move file {file} to '
                         f'{Path(MASTER_ANALYSES, file.name)}:', e)

        time.sleep(0.25)

    logger.info(f'Successfully submitted {len(successful_copies)} '
                f'analyses: ', successful_copies)

    history_df.to_csv(history_file, index=False)


if __name__ == '__main__':
    lock = Path(LOCKFILE)

    breakpoint()

    if lock.exists():
        print('Lockfile exists, exiting')
        sys.exit(0)

    with open(lock, 'w') as lf:
        lf.write('BUSY!\n')

    main(10)


    lock.unlink(missing_ok=True)