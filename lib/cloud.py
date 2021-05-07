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
from concurrent.futures import ThreadPoolExecutor
import sys
from lib.util import (
    gen_pcd_df,
    gen_mesh,
    fmt2regex,
    find_matching_files,
    k2f,
    f2k,
    ls_recursive,
    process_requires,
    source_keys_conv,
    process_file_entries,
    compress_8bit
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

        inrows = inrows.astype(str)
        outfile = Path(savedir, str(outpattern).format_map(inrows.iloc[0].to_dict()))

        # If the processed file already exists just return it
        if outfile.is_file():
            return outfile

        pcds = []

        genecol = 'gene'
        query = ''

        if 'dots_csv' in inrows['source_key'].values:
            query = 'source_key == "dots_csv"'
            genecol = 'gene'
        elif 'dots_csv_unseg' in inrows['source_key'].values:
            query = 'source_key == "dots_csv_unseg"'
            genecol = 'geneID'

        infiles = inrows.query(query)[['channel', 'local_filename']]

        for chan, csv in infiles.values:
            pcd_single = pd.read_csv(csv)
            pcd_single['channel'] = chan
            pcds.append(pcd_single)

        pcds_combined = pd.concat(pcds)
        del pcds

        gen_pcd_df(pcds_combined, genecol=genecol, outfile=outfile)
        del pcds_combined

        return outfile


class DotDetectionPreupload:

    @staticmethod
    def compress_raw_im(
        inrow,
        outpattern,
        savedir
    ):
        if not inrow:
            return None

        im = inrow['filename']
        outfile = Path(savedir, outpattern.format_map(inrow))

        if outfile.is_file():
            return outfile.relative_to(savedir)

        compress_8bit(im, 'DEFLATE', outfile)

        return Path(outfile).relative_to(savedir)


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
            self.generator_class = getattr(sys.modules.get(__name__),
                                           self.config['generator_class'])
        # Same for the preupload function class
        self.preupload_class = None
        if 'preupload_class' in self.config.keys():
            self.preupload_class = getattr(sys.modules.get(__name__),
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
            raise FileNotFoundError(f'master_root specified as '
                                    f'{self.master_root} does not exist')

        self.raw_master_root = config.get('raw_master_root')
        if not Path(self.master_root).is_dir():
            raise FileNotFoundError(f'raw_master_root specified as '
                                    f'{self.raw_master_root} does not exist')

        self.sync_folder = config.get('sync_folder', 'monitoring/')
        Path(self.sync_folder).mkdir(parents=True, exist_ok=True)

        self.analysis_folder = config.get('analysis_folder', 'analyses/')
        self.raw_folder = config.get('raw_folder', 'raw/')

        self.local_store = config.get('local_store', 'webfish_data/')
        self.preupload_root = config.get('preupload_root')

        self.all_datasets = pd.DataFrame()
        self.all_raw_datasets = pd.DataFrame()

        dr = config.get('dataset_root', '')
        self.dataset_root = dr
        # How many levels do we have to fetch to reach the datasets?
        self.dataset_nest = len(Path(dr).parts) - 1

        dre, drglob = fmt2regex(dr)
        self.dataset_re = dre
        self.dataset_glob = drglob

        self.dataset_fields = list(dre.groupindex.keys())

        rr = config.get('raw_dataset_root', '')
        self.raw_dataset_root = rr
        self.raw_nest = len(Path(rr).parts) - 1

        rre, rrglob = fmt2regex(rr)
        self.raw_re = rre
        self.raw_glob = rrglob

        self.raw_fields = list(rre.groupindex.keys())

        self.all_fields = self.dataset_fields + self.raw_fields

        self.pages = {name: Page(config, name) for name in config.get('pages', {})}
        self.pagenames = tuple(self.pages.keys())

        self.bucket_name = config.get('bucket_name')

        pats = []
        for p in self.pages.values():
            pats.extend(list(p.input_patterns.values()))

        with open(Path(self.sync_folder, 'input_patterns'), 'w') as sp:
            sp.write('\n'.join(pats))

    def get_source_datasets(
        self,
        folders=None,
        sync=True
    ):
        possible_folders, fields = find_matching_files(
            self.master_root,
            self.dataset_root)
        fields['folder'] = possible_folders

        all_datasets = pd.DataFrame(fields)

        if folders:
            folders = [Path(f) for f in folders]
            all_datasets = all_datasets.query('folder in @folders').copy()

        self.all_datasets = all_datasets

        all_datasets_file = Path(self.sync_folder, 'all_datasets.csv')

        self.all_datasets.to_csv(all_datasets_file, index=False)

        if sync:
            self.client.client.upload_file(
                str(all_datasets_file),
                Bucket=self.bucket_name,
                Key=str(all_datasets_file)
            )

        return self.all_datasets

    def get_raw_datasets(
            self,
            folders=None,
            sync=True
    ):

        possible_folders, fields = find_matching_files(
            self.raw_master_root,
            self.raw_dataset_root)
        fields['folder'] = possible_folders

        all_raw_datasets = pd.DataFrame(fields)

        if folders:
            folders = [Path(f) for f in folders]
            all_raw_datasets = all_raw_datasets.query('folder in @folders').copy()

        self.all_raw_datasets = all_raw_datasets

        all_raw_datasets_file = Path(self.sync_folder, 'all_raw_datasets.csv')

        self.all_raw_datasets.to_csv(all_raw_datasets_file, index=False)

        if sync:
            self.client.client.upload_file(
                str(all_raw_datasets_file),
                Bucket=self.bucket_name,
                Key=str(all_raw_datasets_file)
            )

        return self.all_raw_datasets

    def find_page_files(
        self,
        page,
        source_folders=None,
        raw_folders=None,
        sync=True
    ):
        """
        find_datafiles
        ------------
        Searches the supplied bucket for top-level folders, which should
        represent available datasets. Searches each of these folders for position
        folders, and each of these for channel folders.

        """

        if self.all_datasets.empty:
            self.get_source_datasets()
            self.get_raw_datasets()

        source_datasets = pd.DataFrame()
        raw_datasets = pd.DataFrame()

        sourcefile_df = pd.DataFrame(columns=self.dataset_fields +
                                   ['folder', 'source_key'])

        rawfile_df = pd.DataFrame(columns=self.raw_fields +
                                  ['folder', 'source_key'])

        # If this page doesn't require any files, just show all datasets
        if not self.pages[page].input_patterns:
            page_datasets = self.all_datasets

        all_sourcefiles = [self.find_source_files(key, pattern, source_folders)
                           for key, pattern in self.pages[page].source_patterns.items()]

        all_rawfiles = [self.find_raw_files(key, pattern, raw_folders)
                        for key, pattern in self.pages[page].raw_patterns.items()]

        if all_sourcefiles:
            sourcefile_df = pd.concat(all_sourcefiles).sort_values(by=self.dataset_fields)
            del all_sourcefiles
            source_datasets = self.filter_datasets(
                sourcefile_df,
                self.dataset_fields,
                self.dataset_root
            )

        if all_rawfiles:
            rawfile_df = pd.concat(all_rawfiles).sort_values(by=self.raw_fields)
            del all_rawfiles
            raw_datasets = self.filter_datasets(
                rawfile_df,
                self.raw_fields,
                self.raw_dataset_root
            )

        self.pages[page].datafiles = pd.concat([sourcefile_df, rawfile_df])
        self.pages[page].datasets = pd.concat([source_datasets, raw_datasets])

        if sync:
            self.save_and_sync(page)

        return self.pages[page].datafiles, self.pages[page].datasets

    @staticmethod
    def filter_datasets(
        datafile_df,
        fields,
        dataset_root
    ):
        page_datasets = []

        for group, rows in datafile_df.groupby(fields):
            dataset = {field: value for field, value
                       in zip(fields, group)}

            dataset['folder'] = dataset_root.format_map(dataset)

            # FIXME: Technically we should also group by the source file variables, but
            #   usually all the positions etc WITHIN one analysis ALL have the same
            #   files present.
            dataset['source_keys'] = list(rows['source_key'].unique())

            # TODO: Add boolean logic to exclude datasets that do not have the right
            #   combo of source keys present. Here we are implicitly keeping any
            #   data set that has ANY one source key because it will form a group with
            #   one row in this for loop.

            page_datasets.append(dataset)

        results = pd.DataFrame(page_datasets)

        return results

    def find_source_files(
        self,
        key,
        pattern,
        folders,
    ):
        paths = None
        if folders:
            paths = []
            # We are assuming folders is a list up to dataset_root nesting - potential
            # datasets to look in. This makes a list of glob results looking for
            # pattern from each supplied folder.
            _, glob = fmt2regex(pattern)
            [paths.extend(Path(self.master_root, f).glob(glob)) for f in folders]

        filenames, fields = find_matching_files(
            str(self.master_root),
            str(Path(self.dataset_root, pattern)),
            paths=paths)

        if filenames:
            fields['source_key'] = key
            fields['filename'] = [f.relative_to(self.master_root) for f in filenames]
            return pd.DataFrame(fields, dtype=str)
        else:
            return None

    def find_raw_files(
        self,
        key,
        pattern,
        folders,
    ):
        paths = None
        if folders:
            paths = []
            # We are assuming folders is a list up to dataset_root nesting - potential
            # datasets to look in. This makes a list of glob results looking for
            # pattern from each supplied folder.
            _, glob = fmt2regex(pattern)
            [paths.extend(Path(self.raw_master_root, f).glob(glob)) for f in folders]

        filenames, fields = find_matching_files(
            str(self.raw_master_root),
            str(Path(self.raw_dataset_root, pattern)),
            paths=paths)

        if filenames:
            fields['source_key'] = key
            fields['filename'] = [f.relative_to(self.raw_master_root) for f in filenames]
            return pd.DataFrame(fields, dtype=str)
        else:
            return None

    def save_and_sync(self, page):
        
        page_sync_file = str(self.pages[page].sync_file)
        try:
            current_sync = pd.read_csv(page_sync_file, dtype=str)
            updated_sync = pd.concat([current_sync, self.pages[page].datasets])
            updated_sync.drop_duplicates(subset=['folder'], keep='last', inplace=True, ignore_index=True)
        except FileNotFoundError:
            updated_sync = self.pages[page].datasets

        updated_sync.to_csv(page_sync_file, index=False)

        self.client.client.upload_file(
            page_sync_file,
            Bucket=self.bucket_name,
            Key=page_sync_file
        )

        page_file_table = str(self.pages[page].file_table)
        try:
            current_files = pd.read_csv(page_file_table, dtype=str)
            updated_files = pd.concat([current_files, self.pages[page].datafiles])
            updated_files.drop_duplicates(subset=['filename'], inplace=True, ignore_index=True)
        except FileNotFoundError:
            updated_files = self.pages[page].datafiles

        updated_files.to_csv(page_file_table, index=False)

        self.client.client.upload_file(
            page_file_table,
            Bucket=self.bucket_name,
            Key=page_file_table
        )

    def run_preuploads(
        self,
        pagename,
        file_df=None,
        nthreads=6
    ):
        page = self.pages[pagename]
        if not page.preupload_class:
            return None

        if not any(page.input_preuploads.values()):
            return None

        if file_df is None:
            file_df = page.datafiles

        if file_df.empty:
            return None

        breakpoint()

        keys_with_preuploads = {k for k, v in page.input_preuploads.items() if v}
        rel_files = file_df.query('source_key in @keys_with_preuploads')
        abs_fnames = [str(Path(self.raw_master_root, f))
                      for f in rel_files['filename'].values]
        rel_files['filename'] = abs_fnames

        output_df = file_df.copy()

        for key, filerows in rel_files.groupby('source_key'):
            preupload_func = page.input_preuploads[key]
            out_format = Path(self.raw_dataset_root, page.input_patterns[key])
            out_format = str(out_format.with_name(
                '__'.join([preupload_func.__name__, out_format.name])))

            with ThreadPoolExecutor(max_workers=nthreads) as exe:
                futures = {}

                for row in filerows.to_dict(orient='records'):
                    # row.filename = self.raw_master_root + row.filename
                    # out_format = self.raw_dataset_root + out_format
                    out_dir = Path(self.preupload_root, out_format.format_map(row)).parent
                    out_dir.mkdir(parents=True, exist_ok=True)

                    futures[row['filename']] = exe.submit(
                        preupload_func, row, out_format, self.preupload_root)

                done = 0
                while done < len(futures):
                    sleep(1)
                    if done % 50 == 0:
                        print(f'Done with {done} files out of {len(futures)}')

                    for fname, future in futures.items():
                        if future.done():
                            res = future.result(1)
                            # update the filename
                            output_df.loc[output_df['filename'] == fname, 'filename'] = res
                            done += 1

        return output_df



    def upload_to_s3(
        self,
        pagename,
        request,
        fields
    ):
        """
        upload_to_s3
        ------------
        Uploads a group of requested fields and keys to s3,
        performing preprocessing if specified in the config.

        Analogous to DataClient.request().
        """

        # FIXME: either use pure filenames from the dataset to avoid
        #   asking whether raw or source (we have page.input_preupload and
        #   input_patterns which are combined), or make 2 functions for uploads?
        assert False, 'FIXME'

        if isinstance(fields, str):
            fields = (fields,)

        if isinstance(request, str) or isinstance(request, Path):
            pass # directly upload it

        page = self.pages[pagename]

        if isinstance(request, dict):
            # Query the datafiles index
            query_source = ' and '.join([f'{k} == "{v}"'
                                  for k, v in request.items()
                                  if k in page.source_datafiles.columns])
            query_raw = ' and '.join([f'{k} == "{v}"'
                                         for k, v in request.items()
                                         if k in page.raw_datafiles.columns])

            needed = self.datafiles.query(query)
        else:
            needed = self.datafiles

        if not fields:
            return needed

        results = {field: self._request(field, needed)
                   for field in fields}


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

        self._page = None
        self._pagename = None

        # See properties below: setting self.pagename triggers a property
        # setter that validates the name and then calls the page property
        # setter, passing a Page object.
        self.pagename = pagename

        self.dataset_root = self.page.dataset_root
        self.dataset_re = self.page.dataset_re
        self.dataset_glob = self.page.dataset_glob
        self.dataset_fields = self.page.dataset_fields

        self.raw_dataset_root = self.page.raw_dataset_root
        self.raw_re = self.page.raw_re
        self.raw_glob = self.page.raw_glob
        self.raw_fields = self.page.raw_fields

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
        """
        sync_with_s3
        -----------
        Download the entire contents of the sync_folder from S3 and read in
        the dataset and datafile listing for this page. This tells us what
        datasets are available and what files and fields are available within
        each dataset.
        """
        error = self.client.download_s3_objects(
            self.bucket_name,
            f2k(self.sync_folder))

        if len(error) > 0:
            raise FileNotFoundError(
                f'select_dataset: errors downloading keys:',
                error)

        self.datasets = pd.read_csv(
            self.page.sync_file,
            converters={'source_keys': source_keys_conv}).set_index(self.dataset_fields)
        self.datafiles = pd.read_csv(self.page.file_table, dtype=str)

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

        results = {field: self._request(field, needed, force_download)
                   for field in fields}

        return results

    def _request(
        self,
        field,
        needed,
        force_download
    ):
        # TODO: make this recursive, so that outputs that rely
        #   on other outputs can be generated.
        results = []

        if (field in self.page.source_files.keys()
                or field in self.page.raw_files.keys()):
            files = needed.query('source_key == @field')['filename'].values
            results = [
                self.retrieve_or_download(f, force_download=force_download)
                for f in files
            ]

        elif field in self.page.output_files.keys():
            required_fields = process_requires(
                self.page.output_files[field].get('requires', []))

            required_rows = needed.query('source_key in @required_fields').copy()
            local_filenames = []

            # fetch as many required files as we can
            for row in required_rows.itertuples():
                local_filenames.append(
                    self.retrieve_or_download(
                        row.filename,
                        force_download=force_download
                    )
                )

            # add the local filenames to the dataframe
            required_rows['local_filename'] = local_filenames

            generator = self.page.output_generators[field]

            if generator is not None:
                results = generator(
                    inrows=required_rows,
                    outpattern=Path(self.dataset_root, self.page.output_patterns[field]),
                    savedir=Path(self.page.local_store)
                )
        else:
            raise ValueError(
                f'request for invalid field {field}, valid are {self.page.file_fields}')

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
