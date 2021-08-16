import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import time
import logging
import shutil
import atexit
from logging.handlers import RotatingFileHandler
from pathlib import Path
from argparse import ArgumentParser
SLURM_JOB_ID = os.getenv('SLURM_JOB_ID', None)

"""
download_json_analyses.py
------------------------
This script is intended to be run as a cron job on the HPC or other master
data repository location. It downloads json files from the supplied S3 bucket
and directory, performs some basic validation, then copies well-formed files to
the active directory where Nick's data pipeline scans for new analysis requests.

It keeps a basic record of the analyses that have been submitted in a CSV file
(eventually a database). It also limits the number of analyses that can be submitted
at once, but in the future we will have more validation and security checks.

It takes the location of the live directory for requests, the S3 prefix for the
requests, and the local directory to download them to from environment variables or
sensible defaults, as seen below.
"""

# By default run from the sandbox directory
WF_HOME = os.getenv('WF_HOME', '/home/lombelet/cron/webfish_sandbox/webfish')
os.chdir(WF_HOME)

# The folder to download requests to and save monitoring files to locally
# (NOT the destination for the JSON files to be picked up by the pipeline)
HISTORY_DIR = os.getenv('WF_HISTORY_DIR', 'json_analyses')
# The name of the request history file
HISTORY_FILE = os.getenv('WF_ANALYSIS_HISTORY',
                              'wf_json_analyses_history.csv')

# The prefix (folder) on the S3 bucket that contains analysis requests
S3_PREFIX = os.getenv('WF_S3_ANALYSIS_PREFIX', 'json_analyses/')

# The live folder that is monitored for new analysis requests
# (The destination for the JSON file to be picked up by the pipeline)
MASTER_ANALYSES = os.getenv('WF_JSON_ANALYSES', '/groups/CaiLab/json_analyses_lincoln')
# Updated 8/16/21: Now part of the config file under 'pipeline_json_dir', same attribute name
# in the DataServer object.

LOCKFILE = 'download_json_analyses.lck'

# Append the webfish root to the module search path and import our classes
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


def process_args():
    parser = ArgumentParser()
    parser.add_argument('--allow-duplicates', action='store_true',
                        help='Permit JSON analyses which are identical in '
                             'all given parameters to one in the history file'
                             ' to still run')

    return parser.parse_args()


def init_server():
    """
    init_server
    -----------
    Initialize a lib.server.DataServer instance and an S3 client. Use read_local_sync
    to gather current dataset and file information (replace=True because we are not going
    to check all S3 contents or local files in this script - trusts that upload_datasets.py
    has been running and creating up-to-date monitoring files.)
    """
    config = yaml.load(open('./consts.yml'), Loader=yaml.Loader)

    s3c = S3Connect(config=config)

    dm = DataServer(config=config, s3_client=s3c)
    dm.read_local_sync(replace=True)

    return dm


def read_history_file(fname):
    """
    read_history_file
    -----------------
    Read in the analysis request history record if it exists, otherwise
    make an empty DF that will be saved.
    """
    history_df = pd.DataFrame()

    if Path(fname).is_file():
        try:
            history_df = pd.read_csv(fname, dtype=str)
        except Exception as e:
            # If any problem opening it, just create in anew
            logger.warning(f'Error opening json analyses history file '
                           f'{fname}: {e}')

            history_df = pd.DataFrame()
    else:
        logger.info(f'json analyses history file {fname} '
                    f'does not exist; will create.')

    return history_df


def download_s3_analysis_requests(dm):
    """
    download_s3_analysis_requests
    -----------------------------
    Check S3 for JSON files in the folder given by S3_PREFIX, then download
    them to the local HISTORY_DIR folder.
    """

    s3_files, _ = dm.client.list_objects(
        bucket_name=dm.bucket_name,
        prefix=S3_PREFIX)

    logger.info(f'Listed {len(s3_files)} objects in '
                f'{dm.bucket_name}/{S3_PREFIX}')

    # only apparent json files
    s3_files_filt = [f for f in s3_files if f.suffix == '.json']

    errors = []
    successes = []

    for f in s3_files_filt:

        # Setting prefix=S3_PREFIX causes this function to search in
        # S3_PREFIX / ... for f.name, but download it as just f.name
        # without creating any other directories. So by supplying
        # local_dir=HISTORY_DIR it will download the S3 key
        # {S3_PREFIX}/{f.name} to local directory {HISTORY_DIR}/{f.name}.
        error = dm.client.download_s3_objects(
            dm.bucket_name,
            str(f.name),
            local_dir=HISTORY_DIR,
            prefix=S3_PREFIX
        )

        if len(error) > 0:
            logger.error(f'Error downloading key {f}: {error}')
            errors.append(f)
        else:
            successes.append(f)
            # Delete the files after we download them
            # TODO: Wait until after validation to delete?
            dm.client.client.delete_object(Bucket=dm.bucket_name, Key=str(f))

    return successes, errors


def convert_channel_entry(ent):
    """
    convert_channel_entry
    ---------------------
    The decoding specification in the JSON files can either be
    the string 'across' or the dict {'individual': ['channel1', 'channel2',...]}.
    We want to convert the latter to a string so that it can be stored in the history file
    and recovered. So if we get a dict like that, just return a string with the channel
    numbers separated by a pipe |. If we get a string of that form, split it and return
    the dict that would go into the json file.
    """

    if isinstance(ent, dict):
        channels = ent['individual']

        return '|'.join(channels)

    return ent


def validate_json(fname, history_df, dm, allow_duplicates=False):
    """
    validate_json
    -------------
    Perform various preliminary validations of the JSON files before
    copying them to Nick's live directory.

    """
    with open(fname) as f:
        new_json = json.load(f, parse_int=str, parse_float=str)

    # We pop the "clusters" key because we don't want to store it in the history file,
    # and it is almost always the same every time.
    cluster_info = new_json.pop('clusters')

    assert int(cluster_info['ntasks']) < 4, 'only 1 task allowed'
    assert cluster_info['mem-per-cpu'] == '10G', 'something other than 10G per cpu?'

    # This should contain all possible datasets including raw and already-run analyses
    existing_datasets = dm.all_datasets.copy()

    possible_users = existing_datasets['user'].unique()

    new_personal = new_json['personal']
    new_dataset = new_json['experiment_name']
    # The analysis name will just be the filename of the JSON
    new_analysis = Path(fname).stem

    # Only allow users with existing folders to submit - this means the user
    # must have a personal folder on the CaiLab HPC storage before submitting
    assert new_personal in possible_users, f'Unknown user {new_personal}'

    possible_datasets = existing_datasets.query(
        'user==@new_personal')['dataset'].unique()

    # Only allow requests for datasets which exist under the selected user.
    assert new_dataset in possible_datasets, f'Unknown raw dataset {new_dataset}'

    possible_analyses = existing_datasets.query(
        'user==@new_personal and dataset==@new_dataset')['analysis'].unique()

    # Require the new analysis name to be unique, not identical to an already-run
    # analysis.
    assert new_analysis not in possible_analyses, \
        f'Analysis {new_analysis} already exists for this dataset and user.'

    # If we have a history DF, also make sure that no identically-named analysis
    # has been submitted to this form. This may guard against someone submitting multiple
    # identical analysis names at the same time, but where the names are not identical to
    # one already existing on the HPC. The check against the existing_datasets DF would not
    # catch those.
    if len(history_df) > 0:
        relevant_history = history_df.query(
            'personal==@new_personal and experiment_name==@new_dataset')

        assert new_analysis not in relevant_history['analysis_name'], \
            f'Analysis {new_analysis} already exists for this dataset and user.'

    # Add an analysis_name column for the history file.
    new_json['analysis_name'] = new_analysis

    # Append the new analysis row to the history dataframe
    new_history = pd.concat([history_df,
                             pd.DataFrame([new_json])])

    if 'decoding' in new_history.columns:
        new_history['decoding'] = [convert_channel_entry(e)
                                   for e in new_history['decoding']]

    # This gives us all columns except the user, dataset and analysis names.
    columns_considered = new_history.columns.difference([
        'personal', 'experiment_name', 'analysis_name'])

    # If allow_duplicates is not specified, we reject requests that have
    # totally identical parameters to an analysis run in the history file already.
    # FIXME: This is potentially too stringent, as I *think* it flags analyses that run
    #   more pipeline stages than an existing one but that overlap on the pipeline
    #   stages that they share. (i.e. analysis 1 was just segmentation, analysis 2
    #   was segmentation + dot detection but the segmentation params were the same
    #   as analysis 1).
    duplicates = new_history.duplicated(subset=columns_considered)

    if all(duplicates) and not allow_duplicates:
        dups = new_history.loc[duplicates, 'analysis_name'].values
        raise ValueError(f'Analyses {dups} uses the same parameters '
                         f'as request {new_analysis}.')

    return new_history.reset_index(drop=True)


def atexit_release(lockfile):
    lockfile.unlink(missing_ok=True)


def main(max_copy=10, allow_duplicates=False):
    # Get the DataServer and S3Connect instances
    dm = init_server()

    # Try to use the live directory specified in the config file,
    # fall back to the environment variable (that also specifies a default)
    copy_dest = getattr(dm, 'pipeline_json_dir', MASTER_ANALYSES)

    # Read in the submission history file
    history_file = Path(HISTORY_DIR, HISTORY_FILE)
    history_df = read_history_file(history_file)

    # Download any pending analysis requests from S3 and delete them from there
    new_success, new_fail = download_s3_analysis_requests(dm)

    logger.info(f'Downloaded {len(new_success)} new JSON files')

    if len(new_fail) > 0:
        logger.warning(f'Failed to download {len(new_fail)} new JSON files: {new_fail}')

    to_copy = []

    for new_file in new_success:
        # Validate each JSON file that we have downloaded. If successful,
        # append to to_copy array. Otherwise validate_json raises an exception
        try:
            history_df = validate_json(
                new_file,
                history_df,
                dm,
                allow_duplicates=allow_duplicates
            )

            to_copy.append(new_file)

        except (KeyError, AssertionError, ValueError) as e:
            # These three exceptions arise from lack of critical keys in the
            # JSON, failure of one of the assertions in validate_json, or
            # some other ValueError.
            logger.error(f'JSON file {new_file} was invalid for '
                         f'the following reason: {e}')

    # max_copy is a throttle on the number of analyses we will submit at one time.
    # Right now, we just drop any files that are beyond this. In the future, we would
    # like to add them to a "pending" list or something, and possibly get them next
    # time this script runs if there are less than max_copy other analyses.
    if len(to_copy) > max_copy:
        logger.error(f'Too many ({len(to_copy)}) new analysis '
                     f'files (max set to {max_copy}) - ignoring '
                     f'these files: {to_copy[max_copy:]}')

        # Since we are ignoring files in to_copy[max_copy:], find and drop them
        # from the history directory. Ideally we would wait to delete files until
        # this step, and we wouldn't delete these excess files, allowing them to
        # be attempted on the next run.
        history_df.drop(
            history_df.loc[np.isin(
                history_df['analysis_name'],
                [t.stem for t in to_copy[max_copy:]]
            )].index, inplace=True)

    successful_copies = []

    for file in to_copy[:max_copy]:
        try:
            # Try to copy the local json file to the live directory that
            # submits pipeline jobs.
            shutil.copy(
                Path(file),
                Path(copy_dest, file.name)
            )

            successful_copies.append(file)
        except (PermissionError, OSError) as e:
            logger.error(f'Unable to move file {file} to '
                         f'{Path(copy_dest, file.name)}:', e)

        # Probably not necessary
        time.sleep(0.1)

    logger.info(f'Successfully submitted {len(successful_copies)} '
                f'analyses: {successful_copies}')

    # Finally, save the history file.
    history_df.to_csv(history_file, index=False)


if __name__ == '__main__':
    os.chdir(WF_HOME)

    lock = Path(LOCKFILE)

    logger = logging.getLogger('lib.server')
    logger.setLevel(logging.DEBUG)

    rth = RotatingFileHandler('download_json_analyses.log', maxBytes=2 ** 16, backupCount=4)
    rth.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        f'<pid {os.getpid()}>[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
    rth.setFormatter(formatter)

    logger.addHandler(rth)

    if lock.exists():
        logger.warning('Lockfile exists, exiting')
        sys.exit(0)

    with open(lock, 'w') as lf:
        lf.write('BUSY!\n')

    atexit.register(atexit_release, lock)

    args = process_args()

    main(10, args.allow_duplicates)

    lock.unlink(missing_ok=True)

