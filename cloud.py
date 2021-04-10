import numpy as np
import pandas as pd
import boto3
import botocore.exceptions as boto3_exc
import json
import yaml
import os
import re
from fnmatch import fnmatch
import configparser as cfparse
from collections.abc import Iterable
from collections import defaultdict
from util import (
    populate_files, 
    populate_genes,  
    populate_mesh,
    safe_join, 
    mesh_from_json,
    gen_pcd_df,
    gen_mesh
)

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
        
        self.local_store = self.config.get('local_store', 'webfish_data/')
        
        if not os.path.isdir(self.local_store):
            os.makedirs(self.local_store)
        
        if bucket_name is None:
            self.bucket_name = self.config['bucket_name']
            
        self.datasets = None
        self.datafiles = None
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
            
    def localpath(
        self,
        key,
        delimiter='/'
    ):
        key = key.removeprefix(self.local_store + delimiter)
        return safe_join(
            os.path.sep,
            [self.local_store, key.replace(delimiter, os.path.sep)]
        )
    
    
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
        """
        _, possible_folders = self.client.grab_bucket( 
            self.bucket_name,
            delimiter=delimiter,
            prefix=prefix,
            recursive=False
        )
        
        datafiles = []
        
        pos_re = re.compile(re.escape(self.config['position_prefix']) + '(\d+)')
        chan_re = re.compile(re.escape(self.config['channel_prefix']) + '(\d+)')
        
        for folder in possible_folders:
            
            # we want to make as few requests to AWS as possible, so it is
            # better to list ALL the objects and filter to find the ones we 
            # want. I think!
            f_all, _ = self.client.grab_bucket(
                self.bucket_name,
                delimiter=delimiter,
                prefix=folder,
                recursive=True
            )
            
            #positions = populate_files(positions, config['position_prefix'])
            rel_files = []
            basenames = []
            positions = []
            channels = []
            downloaded = []
            
            img_pat = safe_join(delimiter, [folder, self.config['img_pattern']])
            csv_pat = safe_join(delimiter, [folder, self.config['csv_pattern']])
            
            
            for f in f_all:
                
                if (fnmatch(f, img_pat)
                 or fnmatch(f, csv_pat)):
                    rel_files.append(f)
                    basenames.append(os.path.basename(f))
                    
                    m = pos_re.search(f)
                    positions.append(m.group(1))
                    
                    m = chan_re.search(f)
                    if m is not None:
                        channels.append(m.group(1))
                    else:
                        channels.append(-1)
                        
                    key2file = self.localpath(f, delimiter=delimiter)
                    downloaded.append(os.path.exists(key2file))
            
            
            datafiles.append(pd.DataFrame({
                'dataset': folder,
                'downloaded': downloaded,
                'file': rel_files,
                'basename': basenames,
                'position': positions,
                'channel': channels
            }))
        
        # one could imagine this table is stored on the cloud and updated every
        # time a dataset is added, then we just need to download it and check
        # our local files.
        self.datafiles = pd.concat(datafiles)
        self.datafiles.to_csv(self.localpath('wf_datafiles.csv'), index=False)
        
        self.datasets = defaultdict(dict)
        
        for (name, pos), grp in self.datafiles.groupby(['dataset', 'position']):
            root_dir = os.path.commonpath(list(grp['file'].values))
            
            mesh = safe_join(
                os.path.sep,
                [self.local_store, root_dir, self.config['mesh_name']]
            )
            
            pcd = safe_join(
                os.path.sep,
                [self.local_store, root_dir, self.config['pcd_name']]
            )
            
            self.datasets[name][pos] = {
                'rootdir': root_dir,
                'meshfile': mesh,
                'meshexists': os.path.isfile(mesh),
                'pcdfile': pcd,
                'pcdexists': os.path.isfile(pcd)
            }
            
        
        json.dump(
            self.datasets,
            open(os.path.join(self.local_store, 'wf_datasets.json'), 
                 'w'
            )
        )
        
           
    def select_dataset(
        self,
        name,
    ):
        """
        select_dataset
        --------------
        
        Prepares to load a dataset. Download all needed files if they are not
        already present. **Always call this BEFORE select_position**
        """
        self.active_dataset_name = name
        self.active_dataset = self.datasets.get(name, None)
        self.active_datafiles = self.datafiles.query('dataset == @name')
        
        needed_files = self.active_datafiles.query(
            'downloaded == False'
        )['file'].values
                
        for f in needed_files:
            errors = self.client.download_s3_objects(
                self.bucket_name,
                f,
                local_dir=self.local_store
            )
            
            if len(errors) > 0:
                raise FileNotFoundError(
                    f'select_dataset: errors downloading keys:', 
                    errors
                )
        
        # update file index with new downloads
        self.datafiles.loc[
            np.isin(self.datafiles['file'], needed_files),
            'downloaded'
        ] = True

        return self.active_dataset, self.active_datafiles
    
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
        
        
        ##### Point cloud (PCD) (dots) processing #####
        if not self.active_position['pcdexists']:
            # we need to read in all the channels' dots CSVs
            channel_dots_files = cur_pos_files.query(
                f'basename == {self.config["csv_name"]}'
            )
            
            pcds = []
            channels = []
            
            for _, row in channel_dots_files.iterrows():
                pcd_single = pd.read_csv(self.localpath(row['file']))
                channel = row['channel']
                pcd_single['channel'] = channel
                
                pcds.append(pcd_single)
                channels.append(channel)
            
            pcds_combined = pd.concat(pcds)
            del pcds
            
            pcds_processed = gen_pcd_df(
                pcds_combined,
                outfile=self.localpath(self.active_position['pcdfile'])
            )
            
            # update every copy of the active position...?
            self.active_position['pcdexists'] = True
            self.active_dataset[position]['pcdexists'] = True
            self.datasets[self.active_dataset_name][position]['pcdexists'] = True
            
        else:
            pcds_processed = pd.read_csv(
                self.localpath(self.active_position['pcdfile'])
            )
            
            channels = list(pcds_processed['channel'].unique())
            
        ##### Mesh processing #####
        if not self.active_position['meshexists']:
            
            # find the image file
            im = cur_pos_files.query(
                f'basename == {self.config["img_name"]}'
            )['file'].values[0]
            im = self.localpath(im)
            print(f'im is {im}')
            
            # generate the mesh from the image
            mesh = mesh_from_json(gen_mesh(
                im,
                separate_regions=False,
                region_data=None,
                outfile=self.localpath(self.active_position['meshfile'])
            ))
            
            # update every copy of the active position...?
            self.active_position['meshexists'] = True
            self.active_dataset[position]['meshexists'] = True
            self.datasets[self.active_dataset_name][position]['meshexists'] = True
            
        else:
            # read the mesh in
            mesh = mesh_from_json(self.localpath(self.active_position['meshfile']))
            
        self.active_mesh = mesh
        self.active_dots = pcd_processed
        
        self.possible_channels = channels
        self.possible_genes = { 
            c: populate_genes(d)
            for c, d in self.activedots.groupby(['channel'])
        }
        
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
        region_name=None
    ):

        if key_id is None or secret_key is None:
            try:
                # Find the name of the credential file from the environment
                cred_file = os.environ.get(config['credentials'], 
                                           os.path.expanduser('~/.aws/credentials'))

                cf = cfparse.ConfigParser()
                cf.read(cred_file)

                # Find the desired profile section
                profile_name = config.get('cred_profile_name', 'default')
                csec = cf[profile_name]

                # Get the key ID and secret key
                key_id = csec['aws_access_key_id']
                secret_key = csec['aws_secret_access_key']

            except:
                key_id = None
                secret_key = None

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
        recursive=False
    ):
        """
        grab_bucket
        -----------
        Takes an s3 client and a bucket name, fetches the bucket,
        and lists top-level folder-like keys in the bucket (keys that have a '/').

        Note: We set a generic MaxKeys parameter for 5000 keys max! 
        If this is exceeded the "IsTruncated" field will be True in the output.

        Returns: bucket object and alphabetically-sorted list of unique top-level
        folders from the bucket.
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
            files = sorted([ o['Key'] for o in objects['Contents'] ])
        if 'CommonPrefixes' in objects.keys():
            folders = sorted([p['Prefix'] for p in objects['CommonPrefixes']])

        return files, folders


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
        
        bucket = self.s3.Bucket(bucket_name)
        
        files, folders = self.grab_bucket(
            bucket_name,
            delimiter=delimiter,
            prefix=s3_key,
            recursive=recursive
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
                
            if local_dir is None:
                target = obj
            else:
                target = os.path.join(local_dir, obj)

            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))

            if obj.endswith('/'):
                continue
            try:
                bucket.download_file(
                    #bucket_name,
                    obj, 
                    target
                )
            except boto3_exc.ClientError as error:
                errors.append({'key': obj, 'error': error})

        return errors
    