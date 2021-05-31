from pathlib import Path
import pandas as pd
import json
import logging
import lib.preuploaders
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from time import time
from lib.core import FilePatterns
from lib.util import (
    fmt2regex,
    find_matching_files,
    notempty,
    empty_or_false,
    process_file_entries
)

server_logger = logging.getLogger(__name__)
# Suppress tifffile warnings
logging.getLogger('tifffile').addHandler(logging.NullHandler())


class DataServer(FilePatterns):
    """
    DataServer
    -----------
    Class that manages data requests mediated by an s3 bucket and pushes local
    files to same bucket
    * Uses same config for source files as DataManager
    * find_page_files: lists local structure rather than s3
    * publishes CSV file listing directory structure

    TODO: Document this
    TODO: Simplify the processing of source vs raw files, and the
        functions generally in this class.
    """

    def __init__(
        self,
        config,
        s3_client,
    ):
        super().__init__(config)

        self.client = s3_client
        self.has_preupload = []

        for name, loc in self.file_locations.items():
            if not Path(loc['root']).is_dir():
                raise FileNotFoundError(f'master_root specified as '
                                        f'{loc["root"]} does not exist')

            for k, v in self.file_entries[name].items():
                if v['preupload']:
                    preupload = getattr(lib.preuploaders, v['preupload'])
                    self.file_entries[name][k]['preupload'] = preupload
                    self.has_preupload.append(k)

        self.sync_folder = config.get('sync_folder', 'monitoring/')
        Path(self.sync_folder).mkdir(parents=True, exist_ok=True)

        self.preupload_root = config.get('preupload_root')

        self.all_datasets = pd.DataFrame()

        self.bucket_name = config.get('bucket_name')

        self.sync_contents = {
            'all_datasets': Path(self.sync_folder, 'all_datasets.csv'),
            'input_patterns': Path(self.sync_folder, 'input_patterns.json'),
            'timestamp': Path(self.sync_folder, 'TIMESTAMP'),
            's3_keys': Path(self.sync_folder, 's3_keys.json'),
            'sync_file': Path(self.sync_folder, f'sync.csv'),
            'file_table': Path(self.sync_folder, f'files.csv'),
            'pending_uploads': Path(self.sync_folder, f'pending.csv')
        }

        self.local_sync = dict.fromkeys(self.sync_contents.keys(), None)
        self.s3_keys = defaultdict(list)
        self.s3_diff = pd.DataFrame()
        self.new_source_keys = []

        self.datafiles = pd.DataFrame()
        self.datasets = pd.DataFrame()
        self.pending = pd.DataFrame()

        self.have_run_preuploads = False

    def read_local_sync(
        self,
        replace=False,
    ):

        def read_or_empty(fname, **kwargs):
            try:
                d = pd.read_csv(fname, **kwargs)
            except pd.errors.EmptyDataError:
                d = pd.DataFrame()
            return d

        sync_folder_contents = list(Path(self.sync_folder).iterdir())

        for name, item in self.sync_contents.items():

            if isinstance(item, Path) and item in sync_folder_contents:
                if name in ('all_datasets', 'all_raw_datasets'):
                    self.local_sync[name] = pd.read_csv(item, dtype=str)

                elif name == 'input_patterns':
                    local_patterns = json.load(open(item))
                    self.new_source_keys = [
                        k for k, v in self.input_patterns.items()
                        if v not in local_patterns.values()
                    ]

                    self.local_sync[name] = local_patterns
                elif name == 'timestamp':
                    self.local_sync[name] = float(open(item).read().strip())
                elif name == 's3_keys':
                    self.local_sync[name] = json.load(open(item))
                elif name == 'sync_file':
                    self.datasets = read_or_empty(item, dtype=str)
                elif name == 'file_table':
                    self.datafiles = read_or_empty(item, dtype=str)
                elif name == 'pending_uploads':
                    self.pending = read_or_empty(item, dtype=str)
                else:
                    pass

        if replace:
            self.all_datasets = self.local_sync['all_datasets'].copy()

            self.datasets = self.local_sync.get(
                'sync_file', pd.DataFrame()).copy()
            self.datafiles = self.local_sync.get(
                'file_table', pd.DataFrame()).copy()
            self.pending = self.local_sync.get(
                'pending_uploads', pd.DataFrame()).copy()

    def check_s3_contents(
        self,
        use_local=False
    ):
        if use_local:
            if isinstance(self.local_sync['s3_keys'], dict):
                self.s3_keys = self.local_sync['s3_keys']
            else:
                use_local = False

        if not use_local:
            paginator = self.client.client.get_paginator('list_objects_v2')

            raw_pag = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=str(self.raw_folder),
                PaginationConfig=dict(PageSize=10000))

            raw_results = raw_pag.build_full_result()['Contents']
            raw_keys = [
                self._preupload_revert(
                    Path(k['Key']).relative_to(self.raw_folder))
                for k in raw_results
            ]

            self.s3_keys['raw'] = raw_keys

            source_pag = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=str(self.analysis_folder),
                PaginationConfig=dict(PageSize=10000))

            source_results = source_pag.build_full_result()['Contents']
            source_keys = [
                self._preupload_revert(
                    Path(k['Key']).relative_to(self.analysis_folder))
                for k in source_results
            ]

            self.s3_keys['source'] = source_keys

        all_keys = self.s3_keys['raw'] + self.s3_keys['source']

        if self.have_run_preuploads:
            # If we already ran preuploads on this set, our local files will
            # have a new name. we actually need to revert it
            local_filenames = [self._preupload_revert(f)
                                   for f in self.datafiles['filename'].values]
        else:
            local_filenames = self.datafiles['filename'].values

        s3_diff = set(local_filenames) - set(all_keys)

        self.s3_diff = self.datafiles.query('filename in @s3_diff').copy()

    def get_datasets(
        self,
        category,
        folders=None,
    ):
        if category not in self.file_cats:
            raise ValueError(f'category {category} unrecognized, one of '
                             f'{self.file_cats} required')

        location = self.file_locations[category]

        possible_folders, fields, mtimes = find_matching_files(
            location['root'],
            location['dataset_format'])
        fields['folder'] = possible_folders

        all_datasets = pd.DataFrame(fields)

        if folders:
            folders = [Path(f) for f in folders]
            all_datasets = all_datasets.query('folder in @folders').copy()

        self.all_datasets = pd.concat([
            self.all_datasets,
            all_datasets
        ]).reset_index(drop=True)

        return self.all_datasets

    def find_files(
        self,
        source_folders=None,
        raw_folders=None,
        since=0,
    ):
        """
        find_files
        ------------
        Searches for all files required by a specified page.

        """

        if self.all_datasets.empty:
            self.get_source_datasets()

        if self.all_raw_datasets.empty:
            self.get_raw_datasets()

        source_datasets = pd.DataFrame()
        raw_datasets = pd.DataFrame()

        sourcefile_df = pd.DataFrame(columns=self.dataset_fields +
                                   ['folder', 'source_key'])

        rawfile_df = pd.DataFrame(columns=self.raw_fields +
                                  ['folder', 'source_key'])

        all_sourcefiles = [self.find_source_files(key, pattern, source_folders)
                           for key, pattern in self.source_patterns.items()]

        all_rawfiles = [self.find_raw_files(key, pattern, raw_folders)
                        for key, pattern in self.raw_patterns.items()]

        if any(notempty(all_sourcefiles)):
            sourcefile_df = pd.concat(
                all_sourcefiles).sort_values(
                by=self.dataset_fields).reset_index(drop=True)

            del all_sourcefiles
            source_datasets = self.filter_datasets(
                sourcefile_df,
                self.dataset_fields,
                self.dataset_root
            )

        if any(notempty(all_rawfiles)):
            rawfile_df = pd.concat(
                all_rawfiles).sort_values(
                by=self.raw_fields).reset_index(drop=True)

            del all_rawfiles
            raw_datasets = self.filter_datasets(
                rawfile_df,
                self.raw_fields,
                self.raw_dataset_root
            )

        self.datafiles = pd.concat(
            [sourcefile_df, rawfile_df]).reset_index(drop=True)
        self.datasets = pd.concat(
            [source_datasets, raw_datasets]).reset_index(drop=True)

        return self.datafiles, self.datasets

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
            dataset['source_keys'] = '|'.join(list(rows['source_key'].unique()))

            # TODO: Add boolean logic to exclude datasets that do not have the right
            #   combo of source keys present. Here we are implicitly keeping any
            #   data set that has ANY one source key because it will form a group with
            #   one row in this for loop.

            page_datasets.append(dataset)

        results = pd.DataFrame(page_datasets)

        return results

    def find_category_files(
        self,
        category,
        key,
        pattern,
        folders,
    ):
        if category not in self.file_cats:
            raise ValueError(f'category {category} unrecognized, '
                             f'must be one of {self.file_cats}')

        location = self.file_locations[category]

        paths = None
        if folders:
            paths = []
            # We are assuming folders is a list up to dataset_root nesting - potential
            # datasets to look in. This makes a list of glob results looking for
            # pattern from each supplied folder.
            _, glob = fmt2regex(pattern)
            [paths.extend(Path(location['root'], f).glob(glob)) for f in folders]
        elif folders == []:
            return pd.DataFrame()

        filenames, fields, mtimes = find_matching_files(
            str(location['root']),
            str(Path(location['dataset_format'], pattern)),
            paths=paths,
        )

        if filenames:
            fields['source_key'] = key
            fields['mtime'] = mtimes
            fields['filename'] = [f.relative_to(location['root']) for f in filenames]
            return pd.DataFrame(fields, dtype=str)
        else:
            return pd.DataFrame()

    def save_and_sync(
        self,
        timestamp=True,
        patterns=True,
        s3_keys=True,
        upload=True
    ):
        if timestamp:
            now = time()
            with open(self.sync_contents['timestamp'], 'w') as ts:
                ts.write(str(now)+'\n')

        if patterns:
            with open(self.sync_contents['input_patterns'], 'w') as ips:
                json.dump(self.input_patterns, ips)

        if s3_keys and any(self.s3_keys.values()):
            with open(self.sync_contents['s3_keys'], 'w') as s3kf:
                json.dump(self.s3_keys, s3kf)

        all_datasets_file = self.sync_contents['all_datasets']
        all_raw_datasets_file = self.sync_contents['all_raw_datasets']

        self.all_datasets.to_csv(all_datasets_file, index=False)
        self.all_raw_datasets.to_csv(all_raw_datasets_file, index=False)

        if upload:
            self.client.client.upload_file(
                str(all_raw_datasets_file),
                Bucket=self.bucket_name,
                Key=str(all_raw_datasets_file)
            )
            self.client.client.upload_file(
                str(all_datasets_file),
                Bucket=self.bucket_name,
                Key=str(all_datasets_file)
            )

        page_sync_file = self.sync_contents['sync_file']
        try:
            current_sync = self.local_sync.get('sync_file', None)
            updated_sync = pd.concat([current_sync, self.datasets])
        except AttributeError:
            updated_sync = self.datasets

        updated_sync.drop_duplicates(
            subset=['folder'],
            keep='last',
            inplace=True,
            ignore_index=True
        )
        updated_sync.to_csv(page_sync_file, index=False)

        page_file_table = self.sync_contents['file_table']
        try:
            current_files = self.local_sync.get('file_table', None)
            updated_files = pd.concat([current_files, self.datafiles])
        except AttributeError:
            updated_files = self.datafiles

        if self.have_run_preuploads:
            preup_filenames = []

            for row in updated_files.itertuples():
                preup_filenames.append(self._preupload_newname(
                    row.filename, row.source_key
                ))
            updated_files['filename'] = preup_filenames

        updated_files.drop_duplicates(
            subset=['filename'],
            inplace=True,
            ignore_index=True
        )

        updated_files.to_csv(page_file_table, index=False)

        # We don't try to merge the pending file table with the existing one, because
        # upload_to_s3 already does that - it considers any existing pending files as well
        # as the new ones supplied to it. If it does error, the remaining files are still in
        # the pending DF, so we can just write it out and replace the existing file.
        # Also, if it's empty we'll delete the file. We don't upload this to s3.
        page_pending_table = self.sync_contents['pending_uploads']
        if not empty_or_false(self.pending):
            self.pending.to_csv(page_pending_table, index=False)
        else:
            page_pending_table.unlink(missing_ok=True)

        if upload:
            self.client.client.upload_file(
                str(page_sync_file),
                Bucket=self.bucket_name,
                Key=str(page_sync_file)
            )
            self.client.client.upload_file(
                str(page_file_table),
                Bucket=self.bucket_name,
                Key=str(page_file_table)
            )

    def _preupload_newname(
        self,
        oldname,
        source_key
    ):
        if source_key not in self.has_preupload:
            return oldname

        # Remove any existing preupload prefix if present
        oldname = self._preupload_revert(oldname)

        preupload_func = self.input_preuploads[source_key]

        return str(Path(oldname).with_name(
            '__'.join([preupload_func.__name__, Path(oldname).name])
        ))

    @staticmethod
    def _preupload_revert(name):
        if '__' not in Path(name).name:
            return str(name)

        # we return the last element of the split string in case
        # there were multiple prefixes
        return str(Path(name).with_name(
            str(name).split('__')[-1]))

    def run_preuploads(
        self,
        file_df=None,
        nthreads=5
    ):
        if not self.has_preupload:
            return file_df, {}

        if empty_or_false(file_df):
            return file_df, {}

        keys_with_preuploads = self.has_preupload

        file_df = file_df.reset_index(drop=True)

        rel_files = file_df.query('source_key in @keys_with_preuploads').copy()

        output_df = file_df.copy()
        savedir = Path(self.preupload_root)

        errors = {}

        for key, filerows in rel_files.groupby('source_key'):

            errors[key] = []

            if key in self.source_patterns.keys():
                abs_root = self.master_root
                data_root = self.dataset_root
            elif key in self.raw_patterns.keys():
                abs_root = self.raw_master_root
                data_root = self.raw_dataset_root

            # prepend proper absolute root
            filerows['filename'] = [str(Path(abs_root, f)) for f in filerows['filename']]

            preupload_func = self.input_preuploads[key]

            in_format = str(Path(data_root, self.input_patterns[key]))
            out_format = self._preupload_newname(in_format, key)

            with ProcessPoolExecutor(max_workers=nthreads) as exe:
                futures = {}

                for row in filerows.to_dict(orient='records'):
                    old_fname = in_format.format_map(row)
                    new_fname = out_format.format_map(row)

                    parent_dir = Path(savedir, new_fname).parent
                    parent_dir.mkdir(parents=True, exist_ok=True)

                    fut = exe.submit(preupload_func, row, out_format, savedir)
                    futures[fut] = (old_fname, new_fname)

                done = 0

                for fut in as_completed(list(futures.keys()), None):
                    err = None

                    if done % 50 == 0:
                        server_logger.debug(
                            f'run_preuploads: done with {done} files out of {len(futures)}')

                    try:
                        _, err = fut.result(1)
                    except Exception as exc:
                        err = exc

                    old_fname, new_fname = futures[fut]

                    if err:
                        errors[key].append((old_fname, err))
                    else:
                        # update the filename
                        output_df.loc[
                            output_df['filename'] == old_fname, 'filename'] = new_fname

                    done += 1

        self.datafiles = pd.concat(
            [self.datafiles, output_df]).reset_index(drop=True)
        # This drops any duplicates that did not get their filename modified
        self.datafiles.drop_duplicates(
            subset=['filename'], inplace=True, ignore_index=True)
        # This drops the *old* unmodified rows. Note we keep *last* because we concatenate
        # file_df on the end of the current datafiles table. So we are keeping the rows from file_df
        # that match on every column *except* filename - those that got modified filenames.
        self.datafiles.drop_duplicates(
            subset=self.datafiles.columns.difference(['filename']),
            keep='last', inplace=True, ignore_index=True
        )

        self.have_run_preuploads = True

        return output_df, errors

    def upload_to_s3(
        self,
        since=0,
        file_df=None,
        run_preuploads=True,
        do_pending=False,
        do_s3_diff=False,
        progress=0,
        dryrun=False,
        use_s3_only=False,
        nthreads=5
    ):
        """
        upload_to_s3
        ------------
        Uploads a group of requested fields and keys to s3,
        performing preprocessing if specified in the config.

        The criteria to be uploaded is:
        mtime > since OR file in page.pending OR file in page.s3_diff
        OR source_key is new (not present in the saved input_patterns)
        UNLESS use_s3_only == True, in which case we ONLY consider the
        files present locally but not on s3. This is useful for starting
        fresh with no local monitoring files but not uploading everything again.
        """
        if not isinstance(file_df, pd.DataFrame):
            file_df = self.datafiles

        # Filter by last modified time
        if 'mtime' in file_df.columns:
            file_df = file_df.astype({'mtime': float})
            file_df = file_df.query('mtime > @since').copy()

        # Add any pending files if present
        if do_pending:
            local_pending = self.local_sync.get('pending_uploads', pd.DataFrame())
            file_df = pd.concat(
                [self.pending, file_df, local_pending]).reset_index(
                drop=True).drop_duplicates(
                subset='filename', ignore_index=True)

        # Add files from any new source keys (that are present in our
        # current config but not in the locally saved one)
        # Note that s3_diff should catch these too, as these files
        # will not have been uploaded before.
        if self.new_source_keys:
            nsks = self.new_source_keys
            file_df = pd.concat(
                [file_df, self.datafiles.query('source_key in @nsks')]
            ).reset_index(drop=True).drop_duplicates(
                subset='filename', ignore_index=True)

        # If this is specified, we don't care about any of the above criteria.
        # And by definition we need to check the s3_diff.
        if use_s3_only:
            file_df = pd.DataFrame()
            do_s3_diff = True

        # Add files present locally but not on s3
        if do_s3_diff:
            if not empty_or_false(self.s3_diff):
                file_df = pd.concat(
                    [self.s3_diff, file_df]).reset_index(
                    drop=True).drop_duplicates(
                    subset='filename', ignore_index=True)

        if run_preuploads:
            server_logger.info(f'upload_to_s3: running preuploads')
            file_df, errors = self.run_preuploads(file_df=file_df, nthreads=nthreads)

            if any(errors.values()):
                server_logger.warning(
                    f'upload_to_s3: ran into some preupload errors')
                server_logger.debug(errors)

                for key_errs in errors.values():
                    bad_fnames = [e[0] for e in key_errs]
                    file_df.drop(
                        index=file_df.query('filename in @bad_fnames').index,
                        inplace=True
                    )

        if empty_or_false(file_df):
            return file_df, 0

        p = 0
        total = len(file_df)

        file_df = file_df.reset_index(drop=True)

        self.pending = file_df.copy()

        if dryrun:
            return file_df, 0

        server_logger.info(f'upload_to_s3: commencing uploading {total} files')

        for row in file_df.itertuples():
            if progress and p % progress == 0:
                server_logger.info(f'upload_to_s3: {p} files done out of {total}')
            p += 1

            if row.source_key in self.source_files.keys():
                s3_type = 'source'
                root = self.master_root
                key_prefix = self.analysis_folder
            elif row.source_key in self.raw_files.keys():
                s3_type = 'raw'
                root = self.raw_master_root
                key_prefix = self.raw_folder

            if row.source_key in self.has_preupload:
                root = Path(self.preupload_root)

            keyname = Path(key_prefix, row.filename)
            filename = Path(root, row.filename)

            try:
                self.client.client.upload_file(
                    str(filename),
                    Bucket=self.bucket_name,
                    Key=str(keyname)
                )

                self.pending.drop(index=row.Index, inplace=True)
                self.s3_keys[s3_type].append(
                  self._preupload_revert(row.filename))

            except Exception as ex:
                server_logger.warning(f'problem uploading file {row.filename}: {ex}')

        return self.pending, p

