#!/usr/bin/env python
"""
ec2_startup.py
--------------
A script to run **locally** to start an ec2 instance to run webfish.

"""
import boto3
import json
import yaml
from paramiko import SSHClient
from scp import SCPClient

ec2_client = boto3.client('ec2')

# This call constitutes one "reservation" - each reservation could start
# more than 1 instance. 
instance_info = ec2_client.run_instances(
    MinCount=1, 
    MaxCount=1,                    
    LaunchTemplate={
        'LaunchTemplateName': 'nano_basic'
    },
    UserData=open('setup.sh').read()
)

cur_id = instance_info['Instances'][0]['InstanceId']

def get_public_conn(ids):
    if not isinstance(ids, list):
        ids = [ids]
    
    instances = ec2_client.describe_instances(InstanceIds=ids)
    
    return { id: (info['PublicIpAddress'], info['PublicDnsName'])
            for id, info in
                zip(ids, instances['Reservations'][0]['Instances'])
           }

## Later....

# describe_instances nests by reservation then instance.
all_instances = ec2_client.describe_instances()

# so we make a 2d list
instance_ids = [ 
    [ i['InstanceId'] for i in r['Instances'] ]
    for r in all_instances['Reservations'] 
]

# and shutdown in batches by reservation (boto3 docs say breaking up 
# long requests into blocks can be faster)
for r in instance_ids:
    ec2_client.terminate_instances(InstanceIds=r)