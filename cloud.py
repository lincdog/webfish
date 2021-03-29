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
    try:
        # Find the name of the credential file from the environment
        cred_file = os.environ[config['credentials']]

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

    print(key_id, secret_key, endpoint_url)
    s3 = boto3.resource('s3',
                        region_name=region_name,
                        endpoint_url=endpoint_url,
                        aws_access_key_id=key_id,
                        aws_secret_access_key=secret_key
                       )
    
    return s3

def grab_bucket(conn, bucket_name):

    bucket = conn.Bucket(bucket_name)

    objects = list(bucket.objects.all())
    # Unique folders sorted alphabetically
    possible_folders = sorted(list(set([o.key.split('/')[0] for o in objects ])))
    
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
        