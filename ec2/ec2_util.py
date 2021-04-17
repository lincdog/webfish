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
import paramiko
import scp
import pandas as pd
import numpy as np



class Webfish:
    
    def __init__(
        self,
        ip,
        keyfile,
        credfile
    ):
        self.ip = ip
        self.keyfile = keyfile
        self.credfile = credfile
        self.webfish_processes = ''
        
    def prepare_webfish(
        self
    ):
        with EasySSH(
            self.ip, 
            self.keyfile,
            username='ec2-user'
        ) as conn:
            conn.putfile(self.credfile)
        
    def launch_webfish(
        self
    ):
        with EasySSH(
            self.ip, 
            self.keyfile,
            username='ec2-user'
        ) as conn:
            command = ('cd webfish && '
               'env WEBFISH_CREDS=../ec2-readcredentials '
               'WEBFISH_HOST=0.0.0.0 python index.py > webfish.log'
              )
            try:
                conn.exec_command(
                    command, 
                    read_outputs=True,
                    timeout=10
                )
            except:
                pass

            try:
                _, out, _ = conn.exec_command(
                    'pgrep -f python -d " "',
                    read_outputs=True
                )

                self.webfish_processes = out.decode().strip()

            except:
                pass

    def kill_webfish(self):

        if len(self.webfish_processes) == 0:
            return

        with EasySSH(
                self.ip,
                self.keyfile,
                username='ec2-user'
        ) as conn:
            try:
                _, out, _ = conn.exec_command(
                    f'kill {self.webfish_processes}',
                    read_outputs=True
                )
            except:
                pass



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
        
        self.res_id = 0
        self.index = 0
        
        self.templates = pd.DataFrame(columns=['ID', 'Launch template'])
        self.instances = self.refresh()
        
        self.res_id = len(self.descs['Reservations'])
        self.index += len(self.instances)
        
        self.statuses = {}        
        

    def launch_instances(
        self,
        num_instances,
        launch_template=None,
        user_data=None,
        user_data_file=None
    ):
        if launch_template is None:
            launch_template = self.default_launch_template
        
        if bool(user_data) is False:
            if bool(user_data_file) is True \
             and os.path.isfile(user_data_file):
                user_data = open(user_data_file, 'r').read()
            
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
        new_templates = [launch_template] * len(new_ids)
        self.templates = self.templates.append(
            pd.DataFrame({'ID': new_ids, 'Launch template': new_templates}))
                
        self.refresh()
        
                
    
    def refresh(
        self,
        include_all=True
    ):
        new_descs = self.client.describe_instances()
        self.descs = new_descs
        
        new_indices = []
        new_res = []
        new_ids = []
        new_ips = []
        new_dns = []
        new_types = []
        new_statuses = []
               
        
        for rid, res in enumerate(new_descs['Reservations']):
            
            rinsts = res['Instances']
                        
            new_indices.extend(list(range(self.index, self.index+len(rinsts))))
            new_res.extend([self.res_id]*len(rinsts))
            
            desc_keys = {'InstanceId': new_ids,
                        'PublicIpAddress': new_ips,
                        'PublicDnsName': new_dns,
                        'InstanceType': new_types
                         }
            
            for inst in rinsts:
                for key, array in desc_keys.items():
                        array.append(inst.get(key, ''))
            
        self.statuses = self.client.describe_instance_status(
            IncludeAllInstances=True
        )['InstanceStatuses']
        
        
        def parse_status(status):
            state = status['InstanceState']['Name']
            check1 = status['InstanceStatus']['Status']
            check2 = status['SystemStatus']['Status']
            
            if state != 'running':
                return state
            
            if check1 != 'ok':
                return check1
            
            if check2 != 'ok':
                return check2
            
            return 'running'
        
        status_df = pd.DataFrame({
            'ID': [s['InstanceId'] for s in self.statuses ],
            'Status':[ parse_status(s) for s in self.statuses ]      
        })
                    
        refresh_df = pd.DataFrame({
            'Index': new_indices,
            'Reservation': new_res,
            'ID': new_ids,
            'Public IP': new_ips,
            'Public DNS': new_dns,
            'Type': new_types,
        })
                
        if len(self.statuses) > 0:
            self.instances = refresh_df.merge(
                self.templates, 
                on='ID', 
                how='left'
            ).merge(status_df,
                    on='ID',
                    how='left'
            )
        else:
            self.instances = refresh_df
        
        
        return self.instances
    
    
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
        status='running',
        invert=False
    ):
        statuses = [ s.strip() for s in status.split('|') ]
        if invert:
            q = 'not in'
        else:
            q = 'in'
        return self.instances.query(f'Status {q} @statuses')
    
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
        status='running',
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
        
    
    
class EasySSH:
    
    def __init__(
        self,
        addr,
        keyfile=None,
        username=None,
        password=None
    ):
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys(keyfile)
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.client.connect(addr,
                            username=username,
                            password=password,
                            key_filename=keyfile)
        
        self.scp = scp.SCPClient(self.client.get_transport())
        
    
    def exec_command(
        self,
        command,
        read_outputs=False,
        **kwargs
    ):
        
        stdin, stdout, stderr = self.client.exec_command(command, **kwargs)
        
        if read_outputs:
            stdout = stdout.read()
            stderr = stderr.read()
        
        return stdin, stdout, stderr
    
    @property
    def pwd(self):
        _, out, _ = self.exec_command('pwd', read_outputs=True)
        
        return out.strip()
    
    def ls(self, loc=''):
        _, out, _ = self.exec_command(f'ls {loc}', read_outputs=True)
        
        return out.strip()
    
    def putfile(
        self,
        source,
        target='.'
    ):
        recursive = os.path.isdir(source)
        
        self.scp.put(source, remote_path=target, recursive=recursive)
        
    def getfile(
        self,
        source,
        target='.'
    ):
        _, testdir, _ = self.exec_command(f'file {source}', read_outputs=True)
        
        recursive = (testdir.find('directory') > -1)
        
        self.scp.get(source, local_path=target, recursive=recursive)
        
    
    def close(self):
        self.scp.close()
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, trace):
        self.close()
            