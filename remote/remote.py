# Start a configured configured EC2 Instance
# Set up experiment environment
# Run Experiment

# TODO: perhaps schedule (?) the following:
# Retrieve results
# Shutdown EC2 instance

import os
import sys
import boto3

# load config
import yaml
with open("../config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

connection = None

def get_connection():
    ec2 = boto3.client(
        'ec2',
        verify=False,
        endpoint_url=cfg['aws']['endpoint'],
        aws_access_key_id=cfg['aws']['aws_access_key_id'],
        aws_secret_access_key=cfg['aws']['aws_secret_access_key'],
        region_name='europe',
    )
    connection = ec2
    return ec2

def start_instance(instance_id):
    ec2 = connection or get_connection()
    response = ec2.start_instances(InstanceIds=[instance_id])
    print('Start instance response:', response)

def stop_instance(instance_id):
    ec2 = connection or get_connection()
    response = ec2.stop_instances(InstanceIds=[instance_id])
    print('Stop instance response:', response)

def setup_instance(instance_id):
    _exec_shell_script_via_ssl(instance_id, 'remote_setup.sh')
    _configure(instance_id)

def run_experiment(instance_id):
    _exec_shell_script_via_ssl(instance_id, 'remote_run_experiment.sh')

def _exec_shell_script_via_ssl(instance_id, script):
    host = instance_id + cfg['remote']['base_host']
    user = cfg['remote']['user']
    command = 'cat {0} | ssh -t {1}@{2}'
    command = command.format(script, user, host)
    os.system(command)

def _configure(instance_id):
    host = instance_id + cfg['remote']['base_host']
    user = cfg['remote']['user']
    remote_config = cfg.copy()
    remote_config['dataset_dir'] = '/home/{}/datasets'.format(cfg['remote']['user'])
    remote_config['results_dir'] = '/home/{}/results'.format(cfg['remote']['user'])
    remote_config['instance_id'] = instance_id
    with open("config.remote.yml", 'w') as ymlfile:
        yaml.dump(remote_config, ymlfile, default_flow_style=False)
    command = 'scp config.remote.yml {}@{}:evaluate-kmeans-smote/config.yml'
    command = command.format(user, host)
    os.system(command)
    os.remove('config.remote.yml')

def main():
    id = sys.argv[1]
    actions = sys.argv[2:]
    if 'start' in actions:
        start_instance(id)
    if 'setup' in actions:
        setup_instance(id)
    if 'experiment' in actions:
        run_experiment(id)
    if 'stop' in actions:
        stop_instance(id)

if __name__ == "__main__":
    main()
