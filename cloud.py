import numpy as np
import boto3
import json
import yaml
import os
import configparser as cfparse

###### AWS Code #######
# assumes credentials & configuration are handled outside python in .aws directory or environment variables

config_file = 'consts.yml'
config = yaml.load(open(config_file), Loader=yaml.Loader)

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

    s3 = boto3.resource('s3',
                        region_name=region_name,
                        endpoint_url=endpoint_url,
                        aws_access_key_id=key_id,
                        aws_secret_access_key=secret_key
                       )
    
    return s3

def grab_bucket(conn, bucket_name):
    """
    grab_bucket
    -----------
    Takes an s3-like connection and a bucket name, fetches the bucket,
    and lists top-level folder-like keys in the bucket (keys that have a '/').
    
    Returns: bucket object and alphabetically-sorted list of unique top-level
    folders from the bucket.
    """
    bucket = conn.Bucket(bucket_name)

    objects = list(bucket.objects.all())
    # Unique folders sorted alphabetically
    possible_folders = sorted(list(set(
        [o.key.split('/')[0] for o in objects if o.key.find('/') > -1]
    )))
    
    return bucket, possible_folders


def download_s3_folder(bucket, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
    bucket_name: the name of the s3 bucket
    s3_folder: the folder path in the s3 bucket
    local_dir: a relative or absolute directory path in the local file system
    """
    #bucket = s3.Bucket(bucket_name)
    
    for obj in bucket.objects.filter(Prefix=s3_folder):
        if local_dir is None:
            target = obj.key 
        else:
            target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        if obj.key.endswith('/'):
            continue
            
        bucket.download_file(obj.key, target)
        