import os
import configparser as cfparse
import numpy as np
from pathlib import Path, PurePath
from time import sleep

import boto3
import botocore.exceptions as boto3_exc

from lib.util import process_file_entries, fmt2regex
import lib.generators as generators
import lib.preuploaders as preuploaders


class Page:
    """
    Page
    -----
    Data class that represents a page of the web app.
    """

    def __init__(
        self,
        config,
        name
    ):
        self.name = name
        self.config = config['pages'][name].copy()

        self.bucket_name = config['bucket_name']

        self.local_store = Path(config.get('local_store', 'webfish_data/'), name)

        self.sync_file = Path(config.get('sync_folder'), f'{name}_sync.csv')
        self.file_table = Path(config.get('sync_folder'), f'{name}_files.csv')

        dr = config.get('dataset_root', '')
        self.dataset_root = dr
        # How many levels do we have to fetch to reach the datasets?
        self.dataset_nest = len(Path(dr).parts) - 1

        dre, drglob = fmt2regex(dr)
        self.dataset_re = dre
        self.dataset_glob = drglob

        rr = config.get('raw_dataset_root', '')
        self.raw_dataset_root = rr
        self.raw_nest = len(Path(rr).parts) - 1

        rre, rrglob = fmt2regex(rr)
        self.raw_re = rre
        self.raw_glob = rrglob

        self.dataset_fields = list(dre.groupindex.keys())
        self.raw_fields = list(rre.groupindex.keys())

        self.file_fields = self.config.get('variables')

        self.source_files = process_file_entries(
            self.config.get('source_files', {}))

        self.output_files = process_file_entries(
            self.config.get('output_files', {}))

        self.global_files = process_file_entries(
            self.config.get('global_files', {}))

        self.raw_files = process_file_entries(
            self.config.get('raw_files', {}))

        self.file_keys = list(self.source_files.keys()) + \
            list(self.output_files.keys()) + \
            list(self.global_files.keys()) + \
            list(self.raw_files.keys())

        if len(np.unique(self.file_keys)) != len(self.file_keys):
            raise ValueError('File keys must be unique across all '
                             'file classes (source, raw, output, global'
                             ' in one Page object.')

        # Try to grab the generator class object from this module
        self.generator_class = None
        if 'generator_class' in self.config.keys():
            self.generator_class = getattr(generators,
                                           self.config['generator_class'])
        # Same for the preupload function class
        self.preupload_class = None
        if 'preupload_class' in self.config.keys():
            self.preupload_class = getattr(preuploaders,
                                           self.config['preupload_class'])

        # Make convenience dicts for the different fields of each file type
        # source files and preupload functions
        self.source_patterns = {}
        self.source_preuploads = {}
        for k, v in self.source_files.items():
            self.source_patterns[k] = v['pattern']

            if v['preupload']:
                preupload = getattr(self.preupload_class, v['preupload'])
            else:
                preupload = None
            self.source_preuploads[k] = preupload

        # Raw files and preupload functions
        self.raw_patterns = {}
        self.raw_preuploads = {}
        for k, v in self.raw_files.items():
            self.raw_patterns[k] = v['pattern']

            if v['preupload']:
                preupload = getattr(self.preupload_class, v['preupload'])
            else:
                preupload = None
            self.raw_preuploads[k] = preupload

        # Make combined source+raw dicts, because when searching for files
        # to upload we want to go through both of these
        self.input_patterns = self.source_patterns | self.raw_patterns
        self.input_preuploads = self.source_preuploads | self.raw_preuploads
        self.has_preupload = [k for k, v in self.input_preuploads.items() if v]
        self.have_run_preuploads = False

        # Output files and generators
        self.output_patterns = {}
        self.output_generators = {}
        for k, v in self.output_files.items():
            self.output_patterns[k] = v['pattern']

            if v['generator']:
                generator = getattr(self.generator_class, v['generator'])
            else:
                generator = None
            self.output_generators[k] = generator

        self.datafiles = None
        self.datasets = None
        self.pending = None
        self.s3_diff = None


class S3Connect:
    """
    S3Connect
    -----------
    Looks in the config dict for information to create an s3 client
    for data storage and retrieval.
    The location of the credentials file is expected to be
    given in an **environment variable** named in the config `credentials` key.
    If not present, the default location of the AWS CLI credentials file is used.
    This file is expected to be an ini-style config, that is it looks like this:

        [profile-name]
        aws_access_key_id = XXXXXXXX
        aws_secret_access_key = XXXXXXX
        [profile-name-2]
        ....

    The profile name is taken from config `cred_profile_name` key, or 'default'
    if not present.

    Finally, the endpoint URL and region to connect to are supplied in the
    `endpoint_url` and `region_name` keys. If not supplied, boto3 defaults to
    the us-east-1 region of the standard AWS S3 endpoint.
    """

    def __init__(
        self,
        config=None,
        key_id=None,
        secret_key=None,
        endpoint_url=None,
        region_name=None,
        wait_for_creds=True,
        sleep_time=1,
        wait_timeout=90
    ):
        def try_creds_file(file, wait_for_creds):
            cf = cfparse.ConfigParser()
            result = cf.read(file)

            key_id = None
            secret_key = None

            if len(result) == 0:
                if wait_for_creds:
                    return False, None, None
                else:
                    raise FileNotFoundError(
                     f'S3Connect: unable to read credentials file'
                     f' {file} and not waiting for it.'
                    )

            try:
                # Find the desired profile section
                profile_name = config.get('cred_profile_name', 'default')
                csec = cf[profile_name]

                # Get the key ID and secret key
                key_id = csec['aws_access_key_id']
                secret_key = csec['aws_secret_access_key']

            except KeyError:
                key_id = None
                secret_key = None

            return True, key_id, secret_key

        if key_id is None or secret_key is None:
            cred_file = Path(os.environ.get(
                config['credentials'],
                '~/.aws/credentials')).expanduser()

            success, key_id, secret_key = False, None, None

            time_slept = 0

            while success is False:
                success, key_id, secret_key = try_creds_file(cred_file, wait_for_creds)

                if success is True:
                    break
                else:
                    # If wait_for_creds == False and we *didn't* find the file.
                    # try_creds_file will raise an exception. So we only get here
                    # if wait_for_creds == True
                    print(f'S3Connect: waiting for credentials file {cred_file}')
                    sleep(sleep_time)
                    time_slept += sleep_time

                if time_slept > wait_timeout:
                    raise FileNotFoundError(
                        f'S3Connect: no file {cred_file} found '
                        f'within {wait_timeout} seconds'
                    )

        if endpoint_url is None:
            endpoint_url = config.get('endpoint_url', None)

        if region_name is None:
            region_name = config.get('region_name', 'us-east-1')

        self.s3 = boto3.resource(
            's3',
            region_name=region_name,
            endpoint_url=endpoint_url,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key
        )
        self.client = self.s3.meta.client

    def list_objects(
            self,
            bucket_name,
            delimiter='/',
            prefix='',
            recursive=False,
            to_path=True
    ):
        """
        list_objects
        -----------
        Takes an s3 client and a bucket name, fetches the bucket,
        and lists top-level folder-like keys in the bucket (keys that have a '/').

        Note: We set a generic MaxKeys parameter for 5000 keys max!
        If this is exceeded the "IsTruncated" field will be True in the output.

        Returns: bucket object and alphabetically-sorted list of unique top-level
        folders from the bucket. If to_path is True, we return lists of pathlib.PurePath
        objects, replacing delimiter with the standard '/'.
        """

        prefix = str(prefix)

        if prefix != '' and not prefix.endswith(delimiter):
            prefix = prefix + delimiter

        if recursive:
            delimiter = ''

        objects = self.client.list_objects(
            Bucket=bucket_name,
            Delimiter=delimiter,
            Prefix=prefix,
            MaxKeys=5000
        )

        # should probably use a paginator ti handle this gracefully
        assert not objects['IsTruncated'], 'list_objects: query had over 5000 keys, response was truncated...'

        files = []
        folders = []

        if 'Contents' in objects.keys():
            files = sorted([o['Key'] for o in objects['Contents']])
        if 'CommonPrefixes' in objects.keys():
            folders = sorted([p['Prefix'] for p in objects['CommonPrefixes']])

        if to_path:
            files = [PurePath(f) for f in files]
            folders = [PurePath(f) for f in folders]

        return files, folders

    def list_to_n_level(
        self,
        bucket_name,
        prefix='',
        level=0,
        delimiter='/'
    ):
        """
        list_to_n_level
        ---------------
        Finds subfolders up to level N from the s3 storage. Uses a brute-force
        approach of just listing all the objects under prefix in the bucket, then
        post-filtering them for the common prefixes with N occurrences of delimiter.

        Because the API call has a limit to the number of keys it will return, and
        if a bucket has many objects that are deeply nested, this method might become
        inefficient, especially for shallow queries (i.e. listing millions of deeply
        nested objects when we are going to only take the first two levels). Also
        it will require the use of a paginator in list_objects to go through all those
        keys.

        But it is faster for smallish buckets like we have now compared to the below
        recursive version, because the recursive version must make an API call for
        every subfolder of every level.

        Returns: list of prefixes up to level N
        """
        all_objects, _ = self.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=True,
            to_path=False
        )

        level += 1

        prefixes = [delimiter.join(str(obj).split(delimiter)[:level]) + delimiter
                    for obj in all_objects if str(obj).count(delimiter) >= level]

        return list(np.unique(prefixes))

    def list_to_n_level_recursive(
        self,
        bucket_name,
        prefix='',
        level=0,
        delimiter='/',
    ):
        """
        list_to_n_level_recursive
        -------------------------
        Mimics searching N levels down into a directory tree but on
        s3 keys. Note that level 0 is top level of the bucket, so level
        1 is the subfolders of the top level folders, and so on.

        This is recursive, which means it avoids listing **all** objects from
        the root of the bucket, but it is significantly slower for smaller buckets
        because of the need to call the S3 API for each subfolder of each level.

        I think this may help it scale into buckets with many thousands or millions
        of objects that are deeply nested, especially for shallow queries.
        But the alternate version of this list_to_n_level() is faster for a small
        bucket.
        """
        def recurse_levels(result,
                           current_prefix,
                           current_level=0,
                           max_level=level
                           ):
            _, next_level = self.list_objects(
                bucket_name,
                delimiter=delimiter,
                prefix=current_prefix,
                recursive=False,
                to_path=False
            )

            if len(next_level) == 0:
                return

            if current_level >= max_level:
                result.extend(next_level)
                return

            for folder in next_level:
                recurse_levels(result, folder, current_level+1, max_level)

        tree = []
        recurse_levels(tree, prefix, 0, level)

        return tree

    def download_s3_objects(
            self,
            bucket_name,
            s3_key,
            local_dir='.',
            delimiter='/',
            prefix='',
            recursive=False
    ):
        """
        download_s3_objects
        -------------------

        * If s3_key is a file, download it
        * If s3_key is a "folder" (a CommonPrefix), download *only files* within
          it - i.e. not further prefixes (folders), *unless* recursive = True.
        """
        #breakpoint()
        local_dir = Path(local_dir)

        bucket = self.s3.Bucket(bucket_name)

        key_with_prefix = str(PurePath(prefix, s3_key))

        files, folders = self.list_objects(
            bucket_name,
            delimiter=delimiter,
            prefix=key_with_prefix,
            recursive=recursive,
            to_path=False
        )

        if len(files) + len(folders) == 0:
            # using a full key name (i.e. a file) as Prefix results in no
            # keys found. Of course, a nonexistent key name also does.
            objects = [key_with_prefix]
        elif len(files) > 0:
            # with recursive=False, files contains the full keys (files) in the
            # first level under prefix. With recursive=True, files contains all
            # keys under prefix at all levels.
            objects = files
        else:
            # if we only have folders, there is nothing to download.
            return []

        errors = []

        for obj in objects:

            f = PurePath(obj).relative_to(prefix)

            if local_dir is None:
                target = f
            else:
                target = Path(local_dir) / f

            if not target.parent.is_dir():
                target.parent.mkdir(parents=True)

            if obj.endswith('/'):
                continue
            try:
                bucket.download_file(
                    obj,
                    str(target)
                )
            except boto3_exc.ClientError as error:
                errors.append({'key': obj, 'error': error})

        return errors
