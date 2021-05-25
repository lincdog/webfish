import pandas as pd
import logging
from datetime import datetime

from pathlib import Path
from lib.core import Page
from lib.util import (
    f2k,
    k2f,
    source_keys_conv,
    process_requires,
    sanitize
)

client_logger = logging.getLogger(__name__)


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
        if pagename and pagename not in self.pagenames:
            raise ValueError(f'A page name (one of {self.pagenames}) is required.')

        self.config = config
        self.client = s3_client

        self.sync_folder = Path(config.get('sync_folder', 'monitoring/'))
        self.sync_folder.mkdir(parents=True, exist_ok=True)

        self.analysis_folder = config.get('analysis_folder', 'analyses/')
        self.raw_folder = config.get('raw_folder', 'raw/')
        self.bucket_name = config['bucket_name']

        self._page = None
        self._pagename = None

        # See properties below: setting self.pagename triggers a property
        # setter that validates the name and then calls the page property
        # setter, passing a Page object.
        self.pagename = pagename

        if self.page:
            # TODO: we don't actually use any of these right now in the page files.
            #   we *could* use the dataset_root and various source/raw/output fields
            #   to automatically generate selectors for the various fields.
            # e.g. user -> dataset -> analysis then position -> hyb -> ...
            # though the config may not be enough to completely specify the
            # structure and UI we want for each individual case.
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
        if not val:
            self._pagename = None
            self.page = None
        elif val in self.pagenames:
            self._pagename = val
            self.page = Page(self.config, val)
        else:
            raise ValueError(f'A valid page name (one of {self.pagenames}) is required.')

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, val):
        self._page = val

        if isinstance(val, Page):
            self._page.local_store.mkdir(parents=True, exist_ok=True)

    def sync_with_s3(
        self,
        download=True
    ):
        """
        sync_with_s3
        -----------
        Download the entire contents of the sync_folder from S3 and read in
        the dataset and datafile listing for this page. This tells us what
        datasets are available and what files and fields are available within
        each dataset.
        """
        if download:
            error = self.client.download_s3_objects(
                self.bucket_name,
                f2k(self.sync_folder))

            if len(error) > 0:
                raise FileNotFoundError(
                    f'sync_with_s3: errors downloading keys:',
                    error)

        if self.page and self.page.sync_file and self.page.file_table:
            self.datasets = pd.read_csv(
                self.page.sync_file,
                converters={'source_keys': source_keys_conv})

            self.datafiles = pd.read_csv(self.page.file_table, dtype=str)
        else:
            source_file = Path(self.sync_folder, 'all_datasets.csv')
            raw_file = Path(self.sync_folder, 'all_raw_datasets.csv')

            self.datasets = pd.concat([
                pd.read_csv(source_file),
                pd.read_csv(raw_file)
            ]).reset_index(drop=True)

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

            client_logger.debug(f'{request}, {query}, {len(self.datafiles)}')

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

        # Input files: no client side processing required
        if (field in self.page.source_files.keys()
                or field in self.page.raw_files.keys()):
            files = needed.query('source_key == @field')['filename'].values
            results = [
                self.retrieve_or_download(f, field=field, force_download=force_download)
                for f in files
            ]

        # Output files: client-side processing required
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
                        field=row.source_key,
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
        client_logger.debug(f'RETRIEVEORDOWNLOAD: starting {key}')
        error = []

        if self.page is not None:
            if field in self.page.source_patterns.keys():
                prefix = self.analysis_folder
            elif field in self.page.raw_patterns.keys():
                prefix = self.raw_folder

        lp = self.local(k2f(key))

        if force_download or not lp.is_file():
            error = self.client.download_s3_objects(
                self.bucket_name,
                str(key),
                prefix=str(prefix),
                local_dir=self.page.local_store
            )

        if len(error) > 0:
            client_logger.warning(f'ERROR: {error}')
            lp = None

        client_logger.warning(f'RETRIVEORDOWNLOAD: ending {key} after '
                              f'{datetime.now()-now} seconds')

        return lp

