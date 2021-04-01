#!/usr/bin/env python
"""
ec2_startup.py
--------------
A script to run **locally** to start an ec2 instance to run webfish.

"""
import boto3
import json
import os
import time
from math import ceil
import yaml
from paramiko import SSHClient
from scp import SCPClient
import pandas as pd

class EasyEC2:
    """
    EasyEC2
    --------
    A little class to make some common operations of EC2 management 
    simple and easy.
    """
    
    def __init__(
        self,
        keyfile=None,
        profile='default',
        default_launch_template='nano_basic'
    ):
        
        self.session = boto3.session.Session(profile_name=profile)
        self.client = self.session.client('ec2')
        
        self.keyfile = keyfile
        self.default_launch_template = default_launch_template
        
        self.instances = pd.DataFrame(columns=['Index',
                                              'Reservation',
                                              'ID',
                                              'Public IP',
                                              'Public DNS',
                                              'Status',
                                              'Launch template',
                                              ])
        
        self.instance_descs = []
        
        self.res_id = 0
        self.index = 0
        

    def launch_instances(
        self,
        num_instances,
        launch_template=None,
        user_data=None
    ):
        if launch_template is None:
            launch_template = self.default_launch_template
        
        if os.path.isfile(user_data):
            user_data = open(user_data, 'r').read()
            
            if 4*ceil(len(user_data)/3) > 2**14: # base64 limited to 16KB
                raise ValueError(f'File {user_data} would be larger than the'
                                ' 16 KB limit after Base 64 encoding'
                                )
        
        response = self.client.run_instances(
            MinCount=num_instances,
            MaxCount=num_instances,
            LaunchTemplate={
                'LaunchTemplateName': launch_template
            },
            UserData=user_data
        )
        
        new_ids = [ r['InstanceId'] for r in response['Instances'] ]
        new_res = [self.res_id] * len(new_ids)
        self.res_id += 1
        
        new_indices = list(range(0, len(new_ids)))
        self.index += len(new_ids)
        
        new_templates = [launch_template] * len(new_ids)
        
        new_descs = self.client.describe_instances(InstanceIds=new_ids)
        new_descs = new_descs['Reservations'][0]['Instances']
        self.instance_descs.extend(new_info)
        
        new_ips = [desc['PublicIpAddress'] for desc in new_descs ]
        new_dns = [desc['PublicDnsName'] for desc in new_descs ]
        
        new_statuses = self.client.describe_instance_status(InstanceIds=new_ids)
        new_statuses = [s['InstanceState']['Name'] for s in new_statuses]
        
        new_instances = pd.DataFrame({
                'Index': new_indices,
                'Reservation': new_res,
                'ID': new_ids,
                'Public IP': new_ips,
                'Public DNS': new_dns,
                'Status': new_statuses,
                'Launch template': new_templates
            })
        
        self.instances = self.instances.append(new_instances).reset_index(drop=True)
        
    
    def refresh(
        self,
        include_all=True
    ):
        new_statuses = self.client.describe_instance_status(
            IncludeAllInstances=include_all)
        new_statuses = [s['InstanceState']['Name'] for s in new_statuses]
        
        self.instances['Status'] = new_statuses
        
        return new_statuses
    
    
    def get_statuses(
        self,
        ids=None
    ):
        if ids is None:
            statuses = self.instances['Status'].values
        else:
            if not isinstance(ids, list):
                ids = [ids]
            
            statuses = self.instances.query('ID in @ids')['Status'].values
        
        return statuses
    
    def get_by_status(
        self,
        status='running'
    ):
        statuses = [ s.strip() for s in status.split('|') ]
        return self.instances.query('Status in @statuses')
    
    def get_by_index(
        self,
        expr=0
    ):
        
        if isinstance(expr, int):
            expr = f'== @expr'
        elif isinstance(expr, list):
            expr = f'in @expr'
            
        return self.instances.query(f'Index {expr}')
    
    def wait_for_status(
        self,
        status='running'
        ids=None,
        sleep=1,
        timeout=90
    ):
        
        statuses = self.get_statuses(ids)
        
        timeslept = 0
        
        while not np.all(statuses == status):
            time.sleep(sleep)
            timeslept += sleep
            
            if timeslept > timeout:
                return False
            
            statuses = self.get_statuses(ids)
        
        return statuses
        
    
    
    def ssh_connect(
        self,
        id=None,
        keyfile=None,
        
    ):
        self.ssh = paramiko.SSHClient()
        

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