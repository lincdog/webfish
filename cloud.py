import numpy as np
import pandas as pd
import boto3
import botocore.exceptions as boto3_exc
import os
from time import sleep
from pathlib import Path, PurePath
import configparser as cfparse
from collections import defaultdict
import json
import sys
from util import (
    gen_pcd_df,
    gen_mesh,
    fmts2file,
    fmt2regex,
    find_matching_files,
    k2f,
    f2k,
    ls_recursive
)


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
    @staticmethod
    def generate_mesh(
        infiles,
        outfile
    ):
        """
        generate_mesh
        ------------
        """
        if Path(outfile).is_file():
            return outfile

        im = infiles['segmentation'][0] # this should be a length 1 list
        # generate the mesh from the image
        gen_mesh(
            im,
            separate_regions=False,
            region_data=None,
            outfile=outfile)

        return outfile

    @staticmethod
    def generate_dots(
        infiles,
        outfile
    ):
        # If the processed file already exists just return it
        if Path(outfile).is_file():
            return outfile

        pcds = []

        # FIXME: we are currently just assigning channel numbers by the
        #   order that infiles['dots_csv'] has them, not actually from their
        #   file path. We could match the regular expression on the file paths
        #   to find the channel.
        for i, csv in enumerate(infiles['dots_csv']):
            pcd_single = pd.read_csv(csv)
            pcds.append(pcd_single)

        pcds_combined = pd.concat(pcds)
        del pcds

        gen_pcd_df(pcds_combined, outfile=outfile)
        del pcds_combined

        return outfile


class DataServer:
    """
    DataServer
    -----------
    Class that manages data requests mediated by an s3 bucket and pushes local
    files to another s3 bucket.
    Basically the same as DataManager but inverted...
    * Uses same config for source files as DataManager - local_store is different
      and bucket/path for requests vs uploads and dataset listings may be different
    * get_datasets: lists local structure rather than s3
    * publishes json file listing directory structure
    * listen: checks for request file on s3
    * put or upload or respond: uploads local file structure to specified
      cloud key structure
    """

    def __init__(
        self,
        config,
        s3_client,
    ):
        pass


class DataManager:
    """
    DataManager
    --------------
    Class that serves files based on config values. Pulls from
    S3 storage if necessary and performs processing using supplied "generator"
    class.
    """

    _page_properties = ('bucket_name', 'sync_file', 'local_store', 'variables',
                        'dataset_root', 'dataset_nest', 'source_files', 'output_files',
                        'source_patterns', 'output_patterns', 'output_generators',
                        'global_files', 'dataset_re', 'dataset_glob', 'file_fields',
                        'datafiles', 'datasets')

    def __init__(
        self,
        config=None,
        s3_client=None,
        bucket_name=None,
        pagename=None,
        is_local=False
    ):
        self.config = config
        self.client = s3_client
        self.is_local = is_local

        self.sync_folder = config.get('sync_folder', 'monitoring/')
        self.pages = config['pages']
        self.pagenames = tuple(self.pages.keys())
        self.active_page = pagename

        if self.is_local:
            self.master_root = config['master_root']

        self.datasets = []
        self.datafiles = None

        for name, page in self.pages.items():
            local_store = Path(page.get('local_store', f'webfish_data/{name}/'))
            local_store.mkdir(parents=True, exist_ok=True)
            self.pages[name]['local_store'] = local_store

            self.pages[name]['source_files'] = page.get('source_files', {})
            self.pages[name]['output_files'] = page.get('output_files', {})
            self.pages[name]['global_files'] = page.get('global_files', {})

            dr = page.get('dataset_root', '')
            self.pages[name]['dataset_root'] = dr
            # How many levels do we have to fetch to reach the datasets?
            self.pages[name]['dataset_nest'] = len(dr.rstrip('/').split('/')) - 1

            re, glob = fmt2regex(dr)
            self.pages[name]['dataset_re'] = re
            self.pages[name]['dataset_glob'] = glob

            self.pages[name]['file_fields'] = list(page.get('source_files', {}).keys()) + \
                list(page.get('output_files', {}).keys()) + \
                list(page.get('global_files', {}).keys())

            self.pages[name]['source_patterns'] = page.get('source_files', {})

            output_files = page.get('output_files', {})

            self.pages[name]['output_patterns'] = {}
            self.pages[name]['output_generators'] = {}

            # Try to grab the generator class object from this module
            if 'generator_class' in page.keys():
                generator_class = getattr(sys.modules.get(__name__),
                                          page['generator_class'])
            else:
                generator_class = None

            for f, v in output_files.items():
                self.pages[name]['output_patterns'][f] = v['pattern']

                if 'generator' in v.keys():
                    self.pages[name]['output_generators'][f] = getattr(
                        generator_class,
                        v['generator']
                    )

    def __getattr__(self, item):
        if item in type(self)._page_properties:
            if self.active_page:
                return self.pages[self.active_page].get(item, None)
            else:
                return {n: p[item] for n, p in self.pages.items()}
        else:
            try:
                result = self.__getattribute__(item)
            except KeyError:
                raise AttributeError(f'Unknown attribute {item}')
            return result

    def __setattr__(self, key, value):
        if key in type(self)._page_properties:
            if self.active_page:
                self.pages[self.active_page][key] = value
            else:
                self.__dict__[key] = value
        else:
            self.__dict__[key] = value

    @property
    def active_page(self):
        return self._active_page

    @active_page.setter
    def active_page(self, val):
        if val is not None and val not in self.pagenames:
            raise ValueError(f'Attempt to set active_page to invalid value {val}'
                             f'valid names are {self.pagenames}')

        self._active_page = val

    def local(
        self,
        key,
        to_master=False,
        page=None,
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

        if page is None:
            page = self.active_page

        if to_master and self.master_root:
            key = self.master_root / key
        elif not key.is_relative_to(self.pages[page]['local_store']):
            key = self.pages[page]['local_store'] / key

        return Path(key)

    def get_datasets(
        self,
        delimiter='/',
        prefix='',
        progress=False
    ):
        if self.is_local:
            results = {}
            for page in self.pagenames:
                self.active_page = page

                results[page] = self._get_datasets(delimiter, prefix, progress)

            return results
        else:
            return self._get_datasets(delimiter, prefix, progress)

    def _get_datasets(
            self,
            delimiter='/',
            prefix='',
            progress=False
    ):
        """
        get_datasets
        ------------
        Searches the supplied bucket for top-level folders, which should
        represent available datasets. Searches each of these folders for position
        folders, and each of these for channel folders.

        Returns: dictionary representing the structure of all available experiments

        """

        if self.is_local:
            # max_nest = max(self.dataset_nest.values())
            possible_folders = ls_recursive(root=self.master_root,
                                            level=self.dataset_nest,
                                            flat=True)
        else:
            possible_folders = self.client.list_to_n_level_recursive(
                self.bucket_name,
                delimiter=delimiter,
                prefix=prefix,
                level=self.dataset_nest
            )

        print(f'possible_folders: {possible_folders}')

        self.datasets = []
        all_datafiles = []

        if not self.source_files:
            return pd.DataFrame()

        n = 0

        for folder in possible_folders:

            if progress:
                if n % 20 == 0:
                    print(f'get_datasets: finished {n} folders')
                n += 1

            datafiles = []
            dataset = f2k(folder, delimiter).rstrip(delimiter)

            d_match = self.dataset_re.match(dataset)
            if d_match:
                dataset_info = d_match.groupdict()
            else:
                continue

            if self.is_local:
                f_all = None
                folder_prefix = self.master_root

                #source_patterns = {}
                # create a unified source_patterns dict by prefixing pattern names
                # with their parent page using dot notation e.g. datavis.segmentation
                #for pa, ps in self.source_patterns.items():
                #    source_patterns.update({'.'.join([pa, k]): p for k, p in ps.items()})
            else:
                # we want to make as few requests to AWS as possible, so it is
                # better to list ALL the objects and filter to find the ones we
                # want. I think!
                # Note that we have set the MaxKeys parameter to 5000, which is hopefully enough.
                # But in the future we want to use a Paginator in S3Connect to avoid this possibility.
                k_all, _ = self.client.list_objects(
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

                folder_prefix = Path()

            missing_source = 0

            for k, p in self.source_patterns.items():
                filenames, fields = find_matching_files(folder_prefix / folder, p, paths=f_all)

                n_matches = len(filenames)

                # if any of the patterns don't match at all, skip this dataset entirely
                if n_matches == 0:
                    missing_source += 1
                    continue

                fields['filename'] = filenames
                fields['field'] = n_matches * [k]

                fields['analysisid'] = folder

                for dk, v in dataset_info.items():
                    fields[dk] = n_matches * [v]

                datafiles.append(pd.DataFrame(fields))

            # If we found at least 1 of each source file, append this dataset's
            # info to the global datafile and datasets arrays. Otherwise skip it.
            if missing_source < len(self.source_patterns):
                all_datafiles.extend(datafiles)
                self.datasets.append(dataset_info)

        # one could imagine this table is stored on the cloud and updated every
        # time a dataset is added, then we just need to download it and check
        # our local files.
        if datafiles:
            self.datafiles = pd.concat(all_datafiles)
            self.datafiles['page'] = self.active_page
            self.datafiles.to_csv(self.local('wf_datafiles.csv'), index=False)
        else:
            self.datafiles = pd.DataFrame()

        if self.is_local:
            json.dump(self.datasets, open('wf_dataset.json', 'w'))
            #self.client.client.put

        return self.datafiles

    def request(
            self,
            request,
            fields=(),
            force_download=False
    ):
        """
        request
        --------------

        Requests output
        example: dataman.request({
          'dataset': 'linus_data',
          'position': '1'
          }, fields='mesh')

          Returns: dict of form {field: filename}
        """
        if isinstance(fields, str):
            fields = (fields,)

        if self.global_files:
            if all([field in self.global_files.keys() for field in fields]):
                return {
                    field: self.retrieve_or_download(
                        self.global_files[field], force_download=force_download)
                    for field in fields
                }

        if self.datafiles is None:
            return None

        if request:
            # Query the datafiles index
            query = ' and '.join([f'{k} == "{v}"'
                                  for k, v in request.items()
                                  if k in self.datafiles.columns])

            needed = self.datafiles.query(query)
        else:
            needed = self.datafiles

        if not fields:
            return needed

        results = {}

        for field in fields:
            if field in self.source_files.keys():
                files = needed.query('field == @field')['filename'].values
                results[field] = [
                    self.retrieve_or_download(f, force_download=force_download)
                    for f in files
                ]
            elif field in self.output_files.keys():
                required_fields = self.output_files[field].get('requires', [])
                required_rows = needed.query('field in @required_fields')

                required_files = defaultdict(list)
                for row in required_rows[['field', 'filename']].values:
                    required_files[row[0]].append(
                        self.retrieve_or_download(row[1],
                                                  force_download=force_download))

                # call the generating function with args required_files
                generator = self.output_generators[field]
                print(f'generators: {self.output_generators}')
                # FIXME: what if the output file already exists??? generator
                #  needs to check if 'outfile' already exists.
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

                    # FIXME: if all fields in the pattern for this output are not
                    #   present in the original request (e.g. we got more than one
                    #   position by just asking for a dataset) then this errors,
                    #   how to run once for each possible outfile?
                    #   Note in practice we currently only ever uniquely specify
                    #   an outfile, so this error will not affect us at first.
                    results[field] = generator(
                        infiles=required_files,
                        outfile=fmts2file(
                            self.local(self.dataset_root),
                            self.output_patterns[field],
                            fields=request))

            else:
                raise ValueError(f'Request for invalid field {field}, valid'
                                 f' options are {self.file_fields}')

        return results

    def retrieve_or_download(
        self,
        key,
        field=None,
        delimiter='/',
        force_download=False
    ):
        error = []

        lp = self.local(k2f(key, delimiter=delimiter))

        if force_download or not lp.is_file():
            error = self.client.download_s3_objects(
                self.bucket_name,
                f2k(key, delimiter=delimiter),
                local_dir=self.local_store
            )

        if len(error) > 0:
            raise FileNotFoundError(
                f'select_dataset: errors downloading keys:',
                error)

        if field is None:
            return lp

        return {field: lp}


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

        files, folders = self.list_objects(
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
