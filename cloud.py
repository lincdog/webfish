import numpy as np
import boto3
import botocore.exceptions as boto3_exc
import json
import yaml
import os
import configparser as cfparse
from collections.abc import Iterable

from app import config

###### AWS Code #######
# assumes credentials & configuration are handled outside python in .aws directory or environment variables


def s3_connect():
    """
    s3_connect
    -----------
    Looks in the config dict for information to create an s3-style resource
    for data storage. 
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
    
    Returns: boto3.resource object representing the connection.
    """
    
    try:
        # Find the name of the credential file from the environment
        cred_file = os.environ.get(config['credentials'], 
                                   os.path.expanduser('~/.aws/credentials'))

        cf = cfparse.ConfigParser()
        cf.read(cred_file)

        # Find the desired profile section
        if 'cred_profile_name' in config.keys():
            csec = cf[config['cred_profile_name']]
        else:
            csec = cf['default']

        # Get the key ID and secret key
        key_id = csec['aws_access_key_id']
        secret_key = csec['aws_secret_access_key']

    except:
        key_id = None
        secret_key = None

    if 'endpoint_url' in config.keys():
        endpoint_url = config['endpoint_url']
    else:
        endpoint_url = None
        
    if 'region_name' in config.keys():
        region_name = config['region_name']
    else:
        region_name = 'us-east-1'

    s3 = boto3.client(
        's3',
        region_name=region_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key
    )
    
    return s3

def grab_bucket(
    client, 
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
    
    Returns: bucket object and alphabetically-sorted list of unique top-level
    folders from the bucket.
    """
    
    if prefix != '' and not prefix.endswith(delimiter):
        prefix = prefix + delimiter
        
    if recursive:
        delimiter = ''
    
    try:
        objects = client.list_objects(
            Bucket=bucket_name, 
            Delimiter=delimiter,
            Prefix=prefix
        )
    except boto3_exc.ClientError:
        return [], []
    
    
    files = []
    folders = []
    
    if 'Contents' in objects.keys():
        files = sorted([ o['Key'] for o in objects['Contents'] ])
    if 'CommonPrefixes' in objects.keys():
        folders = sorted([p['Prefix'] for p in objects['CommonPrefixes']])
        
    return files, folders
    

def download_s3_objects(
    client,
    bucket_name, 
    s3_key,
    delimiter='/',
    local_dir=None,
    recursive=False
):
    """
    download_s3_objects
    -------------------
    
    * If s3_key is a file, download it
    * If s3_key is a "folder" (a CommonPrefix), download *only files* within
      it - i.e. not further prefixes (folders), *unless* recursive = True.
    """
        
    files, folders = grab_bucket(
        client,
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
            client.download_file(
                bucket_name,
                obj, 
                target
            )
        except boto3_exc.ClientError as error:
            errors.append({'key': obj, 'error': error})
        
    return errors