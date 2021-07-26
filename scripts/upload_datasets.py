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
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', None)

"""
upload_datasets.py
------------------
This script is intended to be run as a cron job on the HPC or other master
data repository location. It scans the master data directories for files that
match any of the patterns given in the config file for Webfish ('consts.yml'),
as well as optionally checking S3 to compare with this list. It then chooses files
to upload to S3 on these conditions:

 - It will usually use os.stat() to compare the modification time of the found
    files with the TIMESTAMP file, which records when this script last ran,
    and only upload files modified since the last run of this script.
 - If --fresh is specified, all existing monitoring files are deleted before
    running the scans. Therefore, unless --use-s3-only is specified, ALL files
    on the master repository will be uploaded anew. This could take a very long
    time.
 - If --check-s3 is specified, it lists ALL keys of data files on the S3 bucket and
    compares these to the found files from the scan. Any files found locally but missing
    from S3 will be added to the upload list regardless of their modification time.
    Note that listing all keys from S3 can take a few minutes.
 - If --use-s3-only is specified, ONLY the above comparison of the S3 content with the
    local found files will determine the upload list - mtime is ignored. This is useful
    for "resetting" the monitoring files, ensuring that the HPC-side monitoring is
    consistent with what is *actually* available on the S3.
 - If --dryrun is specified, the file list determination will take place as above
    but nothing will actually be uploaded. Preupload functions (see below) will also
    not be run. The results of the file determination will be written to a CSV file named
    <PID of process>-files-to-upload.csv.
 - The --max-uploads option sets an upper bound on the number of files that we will 
   attempt to upload in one go. This helps prevent cases where the monitoring files are 
   deleted but not rewritten due to an error, or (possible) when the HPC filesystem daemons
   update the mtime of all files (this seems to happen occasionally and results in this
   program thinking it needs to upload **all** files again). If the number of files to upload
   exceeds the maximum, the --use-s3-only option is set to True so that only the S3 difference
   will be uploaded. 

For any file keys specified in consts.yml that include preupload functions, the
script runs them in parallel using 5 processes by default. Each time it runs, it
attempts to run these functions on the files. It is up to the preupload functions to
quickly return the existing filename if they have already run and the processed file
exists in the preupload root.
"""


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

    parser.add_argument('--max-uploads', action='store', type=int,
                        help='Sets an upper bound on the number of files that will '
                             'be preprocessed and uploaded in one sitting. If the '
                             'number exceeds this, --use-s3-only is set to True and only'
                             'the files present locally but not on S3 are uploaded.')

    return parser.parse_args()


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


def search_and_upload(
        dm,
        mtime,
        use_s3_only=False,
        check_s3=False,
        dryrun=False,
        max_uploads=8000,
):
    """
    search_and_upload
    -----------------
    Searches for relevant files from all pages, determines what to upload
    using mtime and/or S3 contents, and performs the uploads.
    """
    results = {}

    tmp, _ = dm.find_files()
    results['all_files'] = len(tmp)
    del tmp

    # Read in the s3 keys from local monitoring files unless
    # check_s3 is True, in which case list all S3 keys to update the monitoring files.
    dm.check_s3_contents(use_local=(not check_s3))

    dryrun_files, _ = dm.upload_to_s3(
        since=mtime,
        do_pending=True,
        run_preuploads=False,
        do_s3_diff=True,
        use_s3_only=use_s3_only,
        dryrun=True
    )

    if len(dryrun_files) > max_uploads:
        logger.warning(f'Found {len(dryrun_files)} files to process and upload, '
                       f'more than the specified --max-uploads of {max_uploads}. '
                       f'Defaulting to S3 key difference only.')
        use_s3_only = True

    if dryrun:
        dryrun_filename = f'{os.getpid()}-files-to-upload.csv'
        dryrun_files.to_csv(dryrun_filename, index=False)
        logger.info(f'Dry run specified, contents of file_df written to {dryrun_filename}')

        results.update(dict(pending_count=len(dryrun_files),
                            uploaded_count=0))

        return results

    pending, uploaded = dm.upload_to_s3(
        since=mtime,
        do_pending=True,
        run_preuploads=True,
        do_s3_diff=True,
        use_s3_only=use_s3_only,
        progress=100,
        dryrun=False
    )

    results.update(dict(pending_count=len(pending),
                        uploaded_count=uploaded))

    return results


def atexit_release(file):
    file.unlink(missing_ok=True)


def main(args):

    # Initialize the DataServer
    dm = init_server()

    # This is read during read_local_sync from the TIMESTAMP
    # file and should have the last time (in seconds) this script ran.
    mtime = dm.local_sync['timestamp'] or 0

    lock = Path(dm.sync_folder, LOCKFILE)

    # Exit if lockfile is present
    if lock.exists():
        logger.info('Lockfile exists, exiting.')
        return 0

    with open(lock, 'w') as lockfp:
        lockfp.write('{0} {1}'.format(os.getpid(), time.time()))

    atexit.register(atexit_release, lock)

    # If --fresh is specified, delete all local monitoring files except for
    # the lockfile. Also set mtime to 0, ignoring the timestamp of the last
    # time this script ran.
    if args.fresh:
        mtime = 0
        # Delete all monitoring files
        for f in Path(dm.sync_folder).iterdir():
            if f.name != LOCKFILE:
                f.unlink()

    # Search for and upload new files according to the mtime and
    # checking S3 conditions outlined above.
    results = search_and_upload(
        dm,
        mtime,
        use_s3_only=args.use_s3_only,
        check_s3=args.check_s3,
        dryrun=args.dryrun,
        max_uploads=args.max_uploads
    )

    logger.info(f'Results: {results}')

    # Save the updated monitoring information locally and upload
    # it to the S3 monitoring folder, which the webapp uses to determine
    # what datasets and files are available. If --dryrun is specified,
    # do not upload to S3.
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

