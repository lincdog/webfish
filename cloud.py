import numpy as np
import pandas as pd
import boto3
import botocore.exceptions as boto3_exc
import json
import os
import re
from time import sleep
from pathlib import Path, PurePath
import configparser as cfparse
from collections import defaultdict
from util import (
    populate_genes,
    mesh_from_json,
    gen_pcd_df,
    gen_mesh,
    fmt2regex,
    fmts2file,
    find_matching_files,
    k2f,
    f2k
)


class DatavisClient:
    """
    DatavisClient
    -------------
    Client-side class that keeps track of active dataset, position, and other information
    for a single session of the Datavis subapp.
    """

    def __init__(
        self,
    ):
        self.active_datafiles = None
        self.active_dataset = None
        self.active_position = None
        self.active_dataset_name = None
        self.active_position_name = None
        self.possible_channels = None
        self.possible_genes = None
        self.selected_genes = None

        self.active_mesh = None
        self.active_dots = None

    @property
    def state(self):
        """
        :property: state
        ----------------
        JSON serializable structure that summarizes the current state of the
        data manager, and thus the datavis app. Can be stored in the browser
        to persist selections across reloads etc.
        """
        return {
            'active_dataset_name': self.active_dataset_name,
            'active_position_name': self.active_position_name,
            'possible_channels': self.possible_channels,
            'possible_genes': self.possible_genes,
            'selected_genes': self.selected_genes
        }

    def request_dataset(
        self,
        name
    ):
        """
        request_dataset
        ---------------
        Ask the manager for a given dataset.
        """

        if name == self.active_dataset_name:
            return

    def request_position(
        self,
        position
    ):
        pass


class DatavisProcessing:
    """
    DatavisProcessing
    -----------------
    Namespace class used to get generating functions for datasets

    FUNCTION TEMPLATE:
    infiles: dictionary of field: local_filepath for required inputs
    outfile: local file name at which to save the result

    returns: result of the generation. Typically a dataframe or something.
    """
    @classmethod
    def generate_mesh(
        cls,
        infiles,
        outfile
    ):
        """
        generate_mesh
        ------------
        """
        pass

    @classmethod
    def generate_dots(
        cls,
        infiles,
        outfile
    ):
        pass


class DatavisStorage:
    """
    DatavisStorage
    --------------
    Class that serves mesh and dots files to the datavis sub-app. Pulls from
    S3 storage if necessary and performs processing. 
    """

    def __init__(
        self,
        config,
        s3_client,
        bucket_name=None
    ):
        self.config = config
        self.client = s3_client

        self.local_store = Path(self.config.get('local_store', 'webfish_data/'))

        if not self.local_store.is_dir():
            self.local_store.mkdir(parents=True)

        if bucket_name is None:
            self.bucket_name = self.config['bucket_name']

        self.datasets = None
        self.datafiles = None

        self.dataset_root = config.get('dataset_root', '')
        # How many levels do we have to fetch to reach the datasets?
        self.dataset_nest = self.dataset_root.strip('/').split('/').count('/')
        self.source_files = config['source_files']
        self.output_files = config['output_files']

        self.source_patterns = {k: p for k, p in self.source_files.items()}
        self.output_patterns = {k: p['pattern'] for k, p in self.output_files.items()}
        self.output_generators = {k: getattr(DatavisProcessing, g, None)
                                  for k, g in self.output_files.items()}

        self.file_fields = list(self.source_files.keys()) + \
                           list(self.output_files.keys())

    def local(
        self,
        key,
        delimiter='/'
    ):
        """
        local:
        ---------
        Takes an s3-style key with an optional custom delimiter (default '/',
        just like Unix filesystem), and returns the local filename where that
        object will be stored.
        """

        key = k2f(key, delimiter)

        if not key.is_relative_to(self.local_store):
            key = self.local_store / key

        return Path(key)

    def get_datasets(
            self,
            delimiter='/',
            prefix=''
    ):
        """
        get_datasets
        ------------
        Searches the supplied bucket for top-level folders, which should 
        represent available datasets. Searches each of these folders for position
        folders, and each of these for channel folders.
        
        Returns: dictionary representing the structure of all available experiments
        
        TODO: First check for the CSVs that list the status of downloaded files, before
        trying to download from S3.
        """

        self.datasets = defaultdict(dict)

        # TODO: In the future, the possible users/datasets/analyses will be
        #   published by the HPC in a json or similar file and we won't have
        #   to do this.
        possible_folders = self.client.list_to_n_level_recursive(
            self.bucket_name,
            delimiter=delimiter,
            prefix=prefix,
            level=self.dataset_nest
        )

        print(f'possible_folders: {possible_folders}')

        datafiles = []

        for folder in possible_folders:

            # we want to make as few requests to AWS as possible, so it is
            # better to list ALL the objects and filter to find the ones we 
            # want. I think!
            # Note that we have set the MaxKeys parameter to 5000, which is hopefully enough.
            # But in the future we want to use a Paginator in S3Connect to avoid this possibility.
            k_all, _ = self.client.grab_bucket(
                self.bucket_name,
                delimiter=delimiter,
                prefix=folder,
                recursive=True,
                to_path=False
            )
            # we want to have two versions: the list of actual keys k_all,
            # and the PurePath list f for efficient matching and translation
            # to local filesystem conventions
            f_all = [PurePath(k.replace(delimiter, '/')) for k in k_all]

            for k, p in self.source_patterns.items():
                filenames, fields = find_matching_files(folder, p, paths=f_all)

                n_matches = len(filenames)

                fields['dataset'] = n_matches * [folder]
                fields['filename'] = filenames
                fields['field'] = n_matches * [k]

                downloaded = [self.local(f).is_file() for f in filenames]
                fields['downloaded'] = downloaded

                datafiles.append(pd.DataFrame(fields))

        # one could imagine this table is stored on the cloud and updated every
        # time a dataset is added, then we just need to download it and check
        # our local files.
        self.datafiles = pd.concat(datafiles)
        self.datafiles.to_csv(self.local('wf_datafiles.csv'), index=False)

        return self.datafiles

    def request(
            self,
            request,
            fields=[]
    ):
        """
        request
        --------------
        
        Requests output
        example: dataman.request({
          'dataset': 'linus_data',
          'position': '1'
          }, fields='mesh')
        """

        if self.datafiles is None:
            return None

        if isinstance(fields, str):
            fields = [fields]

        # Query the datafiles index
        query = ' and '.join([f'{k} == {v}'
                              for k, v in request.items()
                              if k in self.datafiles.columns])

        query = 'downloaded is False ' + query

        needed = self.datafiles.query(query)

        results = {}

        for field in fields:
            if field in self.source_files.keys():
                files = needed.query('field == @field')['filename'].values
                results[field] = self.retrieve_or_download(files)
            elif field in self.output_files.keys():
                required_fields = self.output_files[field].get('requirements', [])
                required_keys = needed.query(
                    'field in @required_fields')['filename'].values
                required_files = self.retrieve_or_download(
                    required_keys, fields=required_fields)

                # call the generating function with args required_files
                generator = self.output_generators[field]

                if generator is not None:
                    # Set the results for this field as the output of calling
                    # the appropriate generator function with input the dict
                    # of local required files, and output the **populated format
                    # string** supplied from config. Assumes that the request dict
                    # is sufficient to populate this, INCLUDING the dataset root.
                    # In other words, request should contain ALL the fields used
                    # to specify a key or file completely: union of those in
                    # config['dataset_root'] and in the output patterns.
                    # Alternatively, we could use the commonpath from the needed
                    # files.
                    results[field] = generator(
                        infiles=required_files,
                        outfile=fmts2file(
                            self.local(self.dataset_root),
                            self.output_patterns[field],
                            fields=request))

            else:
                raise ValueError(f'Request for invalid field {field}, valid'
                                 f' options are {self.file_fields}')

    def retrieve_or_download(
        self,
        keys,
        fields=None,
        delimiter='/'
    ):
        if isinstance(keys, Path) or isinstance(keys, str):
            keys = [keys]

        errors = []
        localpaths = []

        for f in keys:
            lp = self.local(f)
            if not lp.is_file():

                error = self.client.download_s3_objects(
                    self.bucket_name,
                    f2k(f, delimiter=delimiter),
                    local_dir=self.local_store
                )

                errors.append(error)

            localpaths.append(lp)

        if len(errors) > 0:
            raise FileNotFoundError(
                f'select_dataset: errors downloading keys:',
                errors)

        # update file index with new downloads
        self.datafiles.loc[
            np.isin(self.datafiles['filename'], keys),
            'downloaded'
        ] = True

        if fields is None:
            return localpaths

        if len(fields) == len(keys):
            return {f: p for f, p in zip(fields, localpaths)}

    def select_position(
            self,
            position
    ):
        """
        select_position
        ---------------
        Prepare to actually display a dataset. **Assumes data files are already
        downloaded!** Creates mesh and processed dots file if they don't exist.
        """

        self.active_position_name = position
        self.active_position = self.active_dataset.get(position, None)

        cur_pos_files = self.active_datafiles.query('position == @position')

        updated = False

        ##### Point cloud (PCD) (dots) processing #####
        if not self.active_position['pcdexists']:
            # we need to read in all the channels' dots CSVs
            channel_dots_files = cur_pos_files.query(
                'basename == "{}"'.format(self.config['csv_name'])
            )

            pcds = []
            channels = []

            for _, row in channel_dots_files.iterrows():
                pcd_single = pd.read_csv(self.local(row['file']))
                channel = row['channel']
                pcd_single['channel'] = channel

                pcds.append(pcd_single)
                channels.append(channel)

            pcds_combined = pd.concat(pcds)
            del pcds

            pcds_processed = gen_pcd_df(
                pcds_combined,
                outfile=self.local(self.active_position['pcdfile'])
            )

            # update every copy of the active position...?
            self.active_position['pcdexists'] = True
            self.active_dataset[position]['pcdexists'] = True
            self.datasets[self.active_dataset_name][position]['pcdexists'] = True

            updated = True

        else:
            pcds_processed = pd.read_csv(
                self.local(self.active_position['pcdfile'])
            )

            channels = list(pcds_processed['channel'].unique())

        ##### Mesh processing #####
        if not self.active_position['meshexists']:

            # find the image file
            im = cur_pos_files.query(
                'basename == "{}"'.format(self.config['img_name'])
            )['file'].values[0]
            im = self.local(im)
            print(f'im is {im}')

            # generate the mesh from the image
            mesh = mesh_from_json(gen_mesh(
                im,
                separate_regions=False,
                region_data=None,
                outfile=self.local(self.active_position['meshfile'])
            ))

            # update every copy of the active position...?
            self.active_position['meshexists'] = True
            self.active_dataset[position]['meshexists'] = True
            self.datasets[self.active_dataset_name][position]['meshexists'] = True

            updated = True

        else:
            # read the mesh in
            mesh = mesh_from_json(self.local(self.active_position['meshfile']))

        self.active_mesh = mesh
        self.active_dots = pcds_processed

        self.possible_channels = channels
        self.possible_genes = {
            c: populate_genes(d)
            for c, d in self.active_dots.groupby(['channel'])
        }

        if updated:
            json.dump(
                self.datasets,
                open(self.local('wf_datasets.json'), 'w'),
                default=str
            )

        return {
            'mesh': self.active_mesh,
            'dots': self.active_dots,
            'channels': self.possible_channels,
            'genes': self.possible_genes,
        }


###### AWS Code #######

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
            cred_file = os.environ.get(
                config['credentials'],
                os.path.expanduser('~/.aws/credentials'))

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

    def grab_bucket(
            self,
            bucket_name,
            delimiter='/',
            prefix='',
            recursive=False,
            to_path=True
    ):
        """
        grab_bucket
        -----------
        Takes an s3 client and a bucket name, fetches the bucket,
        and lists top-level folder-like keys in the bucket (keys that have a '/').

        Note: We set a generic MaxKeys parameter for 5000 keys max! 
        If this is exceeded the "IsTruncated" field will be True in the output.

        Returns: bucket object and alphabetically-sorted list of unique top-level
        folders from the bucket. If to_path is True, we return lists of pathlib.PurePath
        objects, replacing delimiter with the standard '/'.
        """

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
        assert not objects['IsTruncated'], 'grab_bucket: query had over 5000 keys, response was truncated...'

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
        it will require the use of a paginator in grab_bucket to go through all those
        keys.

        But it is faster for smallish buckets like we have now compared to the below
        recursive version, because the recursive version must make an API call for
        every subfolder of every level.

        Returns: list of prefixes up to level N
        """
        all_objects, _ = self.grab_bucket(
            bucket_name,
            prefix=prefix,
            recursive=True,
            to_path=False
        )

        level += 1

        prefixes = [delimiter.join(obj.split(delimiter)[:level]) + delimiter
                    for obj in all_objects if obj.count(delimiter) >= level]

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
            _, next_level = self.grab_bucket(
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
            local_dir=None,
            delimiter='/',
            recursive=False
    ):
        """
        download_s3_objects
        -------------------

        * If s3_key is a file, download it
        * If s3_key is a "folder" (a CommonPrefix), download *only files* within
          it - i.e. not further prefixes (folders), *unless* recursive = True.
        """
        local_dir = Path(local_dir)

        bucket = self.s3.Bucket(bucket_name)

        files, folders = self.grab_bucket(
            bucket_name,
            delimiter=delimiter,
            prefix=s3_key,
            recursive=recursive,
            to_path=False
        )

        if len(files) + len(folders) == 0:
            # using a full key name (i.e. a file) as Prefix results in no
            # keys found. Of course, a nonexistent key name also does.
            objects = [s3_key]
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

            f = PurePath(obj.replace(delimiter, '/'))

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
