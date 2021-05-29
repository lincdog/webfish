import pandas as pd
import logging
import lib.generators
from datetime import datetime
from pathlib import Path
from lib.core import FilePatterns
from lib.util import (
    f2k,
    k2f,
    source_keys_conv,
    process_requires,
    sanitize,
    process_file_entries
)

client_logger = logging.getLogger(__name__)


class DataClient(FilePatterns):
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
    ):
        super().__init__(config)
        self.client = s3_client

        self.sync_folder = Path(config.get('sync_folder', 'monitoring/'))
        self.sync_folder.mkdir(parents=True, exist_ok=True)

        self.local_store = Path(config.get('local_store', 'webfish_data/'))

        self.analysis_folder = config.get('analysis_folder', 'analyses/')
        self.raw_folder = config.get('raw_folder', 'raw/')
        self.bucket_name = config['bucket_name']

        self.sync_file = Path(self.sync_folder, 'sync.csv')
        self.file_table = Path(self.sync_folder, 'files.csv')
        self.pending = Path(self.sync_folder, 'pending.csv')

        self.all_datasets_file = Path(self.sync_folder, 'all_datasets.csv')
        self.all_raw_datasets_file = Path(self.sync_folder, 'all_raw_datasets.csv')

        self.datafiles = None
        self.datasets = None
        self.all_datasets = None

        # Output files and generators
        self.output_generators = {}
        for k, v in self.output_files.items():
            if v['generator']:
                generator = getattr(lib.generators, v['generator'])
            else:
                generator = None
            self.output_generators[k] = generator

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

        if self.sync_file.exists() and self.file_table.exists():
            self.datasets = pd.read_csv(
                self.sync_file,
                converters={'source_keys': source_keys_conv})

            self.datafiles = pd.read_csv(self.file_table, dtype=str)
        else:
            self.datasets = pd.DataFrame()
            self.datafiles = pd.DataFrame()

        if self.all_datasets_file.exists() and self.all_raw_datasets_file.exists():
            self.all_datasets = pd.concat([
                pd.read_csv(self.all_datasets_file),
                pd.read_csv(self.all_raw_datasets_file)
            ]).reset_index(drop=True)

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
        if (field in self.source_files.keys()
                or field in self.raw_files.keys()):
            files = needed.query('source_key == @field')['filename'].values
            results = [
                self.retrieve_or_download(f, field=field, force_download=force_download)
                for f in files
            ]

        # Output files: client-side processing required
        elif field in self.output_files.keys():
            required_fields = process_requires(
                self.output_files[field].get('requires', []))

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

            generator = self.output_generators[field]

            if generator is not None:
                results = generator(
                    inrows=required_rows,
                    outpattern=Path(self.dataset_root, self.output_patterns[field]),
                    savedir=Path(self.local_store)
                )
        else:
            raise ValueError(
                f'request for invalid field {field}, valid are {self.file_keys}')

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

        if field in self.source_patterns.keys():
            prefix = self.analysis_folder
        elif field in self.raw_patterns.keys():
            prefix = self.raw_folder
        else:
            prefix = ''

        lp = self.local(k2f(key))

        if force_download or not lp.is_file():
            error = self.client.download_s3_objects(
                self.bucket_name,
                str(key),
                prefix=str(prefix),
                local_dir=self.local_store
            )

        if len(error) > 0:
            client_logger.warning(f'ERROR: {error}')
            lp = None

        client_logger.warning(f'RETRIVEORDOWNLOAD: ending {key} after '
                              f'{datetime.now()-now} seconds')

        return lp

