import numpy as np
import pandas as pd
import boto3
import botocore.exceptions as boto3_exc
import os
from time import sleep
from datetime import datetime
from pathlib import Path, PurePath
import configparser as cfparse
from collections import defaultdict
import json
import sys
from util import (
    gen_pcd_df,
    gen_mesh,
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
     - inrows: Dataframe where each row is a file, includes ALL fields from the file
        discovery methods - from dataset_root and whatever source_patterns are required
     - outpattern: pattern from the config file to specify the output filename - this is
        the dataset_root pattern joined to the output_pattern for this output file.
     - savedir: local folder to store output

    returns: Filename(s)
    """
    @staticmethod
    def generate_mesh(
        inrows,
        outpattern,
        savedir
    ):
        """
        generate_mesh
        ------------
        """

        if inrows.empty:
            return None

        outfile = Path(savedir, str(outpattern).format_map(inrows.iloc[0].to_dict()))

        if outfile.is_file():
            return outfile

        im = inrows.query('source_key == "segmentation"')['local_filename'].values[0]
        # generate the mesh from the image
        gen_mesh(
            im,
            separate_regions=False,
            region_data=None,
            outfile=outfile)

        return outfile

    @staticmethod
    def generate_dots(
        inrows,
        outpattern,
        savedir
    ):
        if inrows.empty:
            return None

        outfile = Path(savedir, str(outpattern).format_map(inrows.iloc[0].to_dict()))

        # If the processed file already exists just return it
        if outfile.is_file():
            return outfile

        pcds = []

        infiles = inrows.query('source_key == "dots_csv"')[['channel', 'local_filename']]
        for chan, csv in infiles.values:
            pcd_single = pd.read_csv(csv)
            pcd_single['channel'] = chan
            pcds.append(pcd_single)

        pcds_combined = pd.concat(pcds)
        del pcds

        gen_pcd_df(pcds_combined, outfile=outfile)
        del pcds_combined

        return outfile


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

        self.source_files = self.config.get('source_files', {})
        self.output_files = self.config.get('output_files', {})
        self.global_files = self.config.get('global_files', {})

        dr = config.get('dataset_root', '')
        self.dataset_root = dr
        # How many levels do we have to fetch to reach the datasets?
        self.dataset_nest = len(dr.rstrip('/').split('/')) - 1

        dre, glob = fmt2regex(dr)
        self.dataset_re = dre
        self.dataset_glob = glob

        self.dataset_fields = list(dre.groupindex.keys())
        self.file_fields = self.config.get('variables')

        self.file_keys = list(self.config.get('source_files', {}).keys()) + \
                           list(self.config.get('output_files', {}).keys()) + \
                           list(self.config.get('global_files', {}).keys())

        self.source_patterns = self.config.get('source_files', {})

        self.output_files = self.config.get('output_files', {})

        self.output_patterns = {}
        self.output_generators = {}

        # Try to grab the generator class object from this module
        if 'generator_class' in self.config.keys():
            self.generator_class = getattr(sys.modules.get(__name__),
                                           self.config['generator_class'])
        else:
            self.generator_class = None

        for f, v in self.output_files.items():
            self.output_patterns[f] = v['pattern']

            if 'generator' in v.keys():
                self.output_generators[f] = getattr(
                    self.generator_class,
                    v['generator']
                )

        self.datafiles = None
        self.datasets = None


class DataServer:
    """
    DataServer
    -----------
    Class that manages data requests mediated by an s3 bucket and pushes local
    files to another s3 bucket.
    Basically the same as DataManager but inverted...
    * Uses same config for source files as DataManager - local_store is different
      and bucket/path for requests vs uploads and dataset listings may be different
    * find_datafiles: lists local structure rather than s3
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
        self.config = config
        self.client = s3_client

        self.master_root = config.get('master_root')
        if not Path(self.master_root).is_dir():
            raise FileNotFoundError(f'master_root specified as {self.master_root} does not exist')

        self.sync_folder = config.get('sync_folder', 'monitoring/')
        Path(self.sync_folder).mkdir(parents=True, exist_ok=True)

        self.analysis_folder = config.get('analysis_folder', 'analyses/')

        self.all_datasets = pd.DataFrame()
        dr = config.get('dataset_root', '')
        self.dataset_root = dr
        # How many levels do we have to fetch to reach the datasets?
        self.dataset_nest = len(Path(dr).parts) - 1

        dre, glob = fmt2regex(dr)
        self.dataset_re = dre
        self.dataset_glob = glob

        self.dataset_fields = list(dre.groupindex.keys())
        self.pages = {name: Page(config, name) for name in config.get('pages', {})}
        self.pagenames = tuple(self.pages.keys())

        self.bucket_name = config.get('bucket_name')

    def get_datasets(
        self,
        folders=None
    ):
        possible_folders = ls_recursive(root=self.master_root,
                                        level=self.dataset_nest,
                                        flat=True)

        if folders:
            folders = [Path(f) for f in folders]
            possible_folders = list(set(folders) & set(possible_folders))

        datasets = []
        dataset_folders = []

        for f in possible_folders:
            f = str(f).rstrip('/')
            d_match = self.dataset_re.match(f)

            if d_match:
                dataset_info = d_match.groupdict()
                dataset_info['folder'] = f
                datasets.append(dataset_info)
                dataset_folders.append(f)

        self.all_datasets = pd.DataFrame(datasets, dtype=str)

        all_datasets_file = Path(self.sync_folder, 'all_datasets.csv')

        self.all_datasets.to_csv(all_datasets_file, index=False)

        self.client.client.upload_file(
            str(all_datasets_file),
            Bucket=self.bucket_name,
            Key=str(all_datasets_file)
        )

        return datasets

    def find_datafiles(
            self,
            page=None,
    ):
        """
        find_datafiles
        ------------
        Searches the supplied bucket for top-level folders, which should
        represent available datasets. Searches each of these folders for position
        folders, and each of these for channel folders.

        Returns: dictionary representing the structure of all available experiments

        """

        if self.all_datasets.empty:
            self.get_datasets()

        page_datasets = []
        datafile_df = pd.DataFrame(columns=self.dataset_fields + ['folder', 'source_keys'], dtype=str)

        if not self.pages[page].source_files:
            page_datasets = self.all_datasets

        all_datafiles = []

        for key, pattern in self.pages[page].source_patterns.items():
            filenames, fields = find_matching_files(str(self.master_root),
                                                    str(PurePath(self.dataset_root, pattern)))

            fields['source_key'] = key
            fields['filename'] = [f.relative_to(self.master_root) for f in filenames]
            all_datafiles.append(pd.DataFrame(fields))

        if all_datafiles:
            datafile_df = pd.concat(all_datafiles).sort_values(by=self.dataset_fields).astype(str)
            del all_datafiles

            for group, rows in datafile_df.groupby(self.dataset_fields):
                dataset = {field: value for field, value in zip(self.dataset_fields, group)}

                dataset['folder'] = self.dataset_root.format(**dataset)

                # FIXME: Technically we should also group by the source file variables, but
                #   usually all the positions etc WITHIN one analysis ALL have the same
                #   files present.
                dataset['source_keys'] = list(rows['source_key'].unique())

                # TODO: Add boolean logic to exclude datasets that do not have the right
                #   combo of source keys present. Here we are implicitly keeping any
                #   data set that has ANY one source key because it will form a group with
                #   one row in this for loop.

                page_datasets.append(dataset)

        page_datasets = pd.DataFrame(page_datasets, dtype=str)

        self.pages[page].datafiles = datafile_df
        self.pages[page].datasets = page_datasets

        page_sync_file = str(self.pages[page].sync_file)
        page_datasets.to_csv(page_sync_file, index=False)

        self.client.client.upload_file(
            page_sync_file,
            Bucket=self.bucket_name,
            Key=page_sync_file
        )

        page_file_table = str(self.pages[page].file_table)
        datafile_df.to_csv(page_file_table, index=False)

        self.client.client.upload_file(
            page_file_table,
            Bucket=self.bucket_name,
            Key=page_file_table
        )

        return page_datasets, datafile_df


class DataClient:
    """
    DataClient
    --------------
    Class that serves files based on config values. Pulls from
    S3 storage if necessary and performs processing using supplied "generator"
    class.
    """

    def __init__(
        self,
        config=None,
        s3_client=None,
        pagename=None,
    ):
        self.pagenames = config['pages'].keys()
        if pagename not in self.pagenames:
            raise ValueError(f'A page name (one of {self.pagenames}) is required.')

        self.config = config
        self.client = s3_client

        self.sync_folder = Path(config.get('sync_folder', 'monitoring/'))
        self.sync_folder.mkdir(parents=True, exist_ok=True)

        self.analysis_folder = config.get('analysis_folder', 'analyses/')
        self.bucket_name = config['bucket_name']

        self.dataset_root = Path(config.get('dataset_root', ''))
        dre, glob = fmt2regex(self.dataset_root)
        self.dataset_re = dre
        self.dataset_glob = glob

        self.dataset_fields = list(dre.groupindex.keys())

        self._page = None
        self._pagename = None
        self.pagename = pagename

        self.datafiles = None
        self.datasets = None

    @property
    def pagename(self):
        return self._pagename

    @pagename.setter
    def pagename(self, val):
        if val not in self.pagenames:
            raise ValueError(f'A valid page name (one of {self.pagenames}) is required.')

        self._pagename = val
        self.page = Page(self.config, val)

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, val):
        self._page = val
        self._page.local_store.mkdir(parents=True, exist_ok=True)

    def sync_with_s3(
        self
    ):
        error = self.client.download_s3_objects(
            self.bucket_name,
            f2k(self.sync_folder),
            local_dir='.'
        )

        if len(error) > 0:
            raise FileNotFoundError(
                f'select_dataset: errors downloading keys:',
                error)

        self.datasets = pd.read_csv(self.page.sync_file).set_index(self.dataset_fields).astype(str)
        self.datafiles = pd.read_csv(self.page.file_table).astype(str)

    def local(
        self,
        key,
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
            page = self.pagename

        if not key.is_relative_to(self.page.local_store):
            key = self.page.local_store / key

        return Path(key)

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

        if self.page.global_files:
            if all([field in self.page.global_files.keys() for field in fields]):
                return {
                    field: self.retrieve_or_download(
                        self.page.global_files[field], force_download=force_download)
                    for field in fields
                }

        if self.datafiles is None:
            return {}

        if request:
            # Query the datafiles index
            query = ' and '.join([f'{k} == "{v}"'
                                  for k, v in request.items()
                                  if k in self.datafiles.columns])

            print(request, query, len(self.datafiles))

            needed = self.datafiles.query(query)
        else:
            needed = self.datafiles

        if not fields:
            return needed

        results = {}

        for field in fields:

            if field in self.page.source_files.keys():
                files = needed.query('source_key == @field')['filename'].values
                results[field] = [
                    self.retrieve_or_download(f, force_download=force_download)
                    for f in files
                ]

            elif field in self.page.output_files.keys():

                required_fields = self.page.output_files[field].get('requires', [])
                required_rows = needed.query('source_key in @required_fields').copy()
                local_filenames = []

                for row in required_rows.itertuples():
                    local_filenames.append(self.retrieve_or_download(
                        row.filename,
                        force_download=force_download))

                required_rows['local_filename'] = local_filenames
                # call the generating function with args required_files
                generator = self.page.output_generators[field]

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
                        inrows=required_rows,
                        outpattern=Path(self.dataset_root, self.page.output_patterns[field]),
                        savedir=Path(self.page.local_store)
                    )
            else:
                raise ValueError(f'Request for invalid field {field}, valid'
                                 f' options are {self.page.file_fields}')

        return results

    def retrieve_or_download(
        self,
        key,
        field=None,
        force_download=False
    ):
        now = datetime.now()
        print(f'RETRIEVEORDOWNLOAD: starting {key}')
        error = []

        lp = self.local(k2f(key))

        if force_download or not lp.is_file():
            error = self.client.download_s3_objects(
                self.bucket_name,
                str(key),
                prefix=str(self.analysis_folder),
                local_dir=self.page.local_store
            )

        if len(error) > 0:
            lp = None

        print(f'RETRIVEORDOWNLOAD: ending {key} after {datetime.now()-now}')

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
