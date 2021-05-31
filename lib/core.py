import os
import configparser as cfparse
import numpy as np
from pathlib import Path, PurePath
from time import sleep

import boto3
import botocore.exceptions as boto3_exc
import jmespath

from lib.util import process_file_entries, process_file_locations, fmt2regex
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

        self.title = self.config.get('title', self.name)
        self.description = self.config.get('description', '')

        self.file_fields = self.config.get('variables')


class FilePatterns:

    def __init__(
        self,
        config
    ):
        self.config = config
        breakpoint()
        self.file_locations = process_file_locations(
            config.get('file_locations', {}))
        self.file_cats = list(self.file_locations.keys())

        self.all_fields = set(jmespath.search(
            '*.dataset_format_fields[]',
            self.file_locations
        ))

        self.file_entries = {
            k: process_file_entries(v)
            for k, v in config.get('file_patterns', {}).items()
        }

        self.file_keys = jmespath.search('map(&keys(@), values(@))[]', self.file_entries)

        if len(set(self.file_keys)) != len(self.file_keys):
            raise ValueError('File keys must be unique across all '
                             'file classes (source, raw, output, global')

    @property
    def file_patterns(self):
        return {c: {k: v['pattern'] for k, v in entry.items()}
                for c, entry in self.file_entries.items()}

    @property
    def input_patterns(self):
        return {k: v for k, v in self.file_patterns.items()
                if k in self.file_cats}

    def category_patterns(self, cat):
        return self.file_patterns[cat]

    def key_info(self, key):
        """
        key_info
        ---------------
        Given a key from the possible file keys given in the config,
        returns all the path components required to localize it.
        """
        if key not in self.file_keys:
            raise ValueError(f'key_to_fullpath: key must be one of {self.file_keys}')

        category = ''
        root = ''
        dataset = ''
        pattern = ''
        prefix = ''
        for name, info in self.file_entries.items():
            if key in info.keys():
                category = name
                root = self.file_locations[name]['root']
                dataset = self.file_locations[name]['dataset_format']
                prefix = self.file_locations[name]['prefix']
                pattern = info['pattern']
                break

        return category, root, dataset, prefix, pattern


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
