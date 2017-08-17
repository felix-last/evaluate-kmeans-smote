# Start a configured configured EC2 Instance
# Set up experiment environment
# Run Experiment
# Notify about completion
# Shutdown instance
# TODO: perhaps schedule (?) the following:
# Retrieve results

import os
import sys
import boto3
import requests
import yaml
import urllib.request
import imbalanced_benchmark

# load config
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

connection = None

# Functions called from workstation.
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

def terminate_instance(instance_id):
    ec2 = connection or get_connection()
    response = ec2.stop_instances(InstanceIds=[instance_id])
    print('Terminate instance response:', response)


def setup_instance(instance_id):
    _exec_shell_script_via_ssl(instance_id, 'remote/setup_instance.sh')
    _configure(instance_id)


def load_dataset(instance_id):
    _exec_shell_script_via_ssl(instance_id, 'remote/load_dataset.sh')


def run_experiment(instance_id):
    _exec_shell_script_via_ssl(instance_id, 'remote/run_experiment.sh')


def retrieve_results(instance_id):
    host = instance_id + cfg['remote']['base_host']
    user = cfg['remote']['user']
    target = './results' # TODO: actually cfg['results_dir']
    command = 'scp -r {}@{}:results/* {}'.format(user, host, target)
    os.system(command)


def _exec_shell_script_via_ssl(instance_id, script):
    host = instance_id + cfg['remote']['base_host']
    user = cfg['remote']['user']
    command = 'cat {0} | ssh -t {1}@{2}'
    command = command.format(script, user, host)
    os.system(command)


def _configure(instance_id, dataset=None):
    host = instance_id + cfg['remote']['base_host']
    user = cfg['remote']['user']
    remote_config = cfg.copy()
    remote_config['dataset_dir'] = '/home/{}/datasets'.format(
        cfg['remote']['user'])
    remote_config['results_dir'] = '/home/{}/results'.format(
        cfg['remote']['user'])
    remote_config['instance_id'] = instance_id
    if dataset is not None:
        remote_config['dataset'] = dataset
    with open("config.remote.yml", 'w') as ymlfile:
        yaml.dump(remote_config, ymlfile, default_flow_style=False)
    command = 'scp config.remote.yml {}@{}:evaluate-kmeans-smote/config.yml'
    command = command.format(user, host)
    os.system(command)
    os.remove('config.remote.yml')


# Functions called from the remote instance.
def _run_experiment():
    """
    Executed locally from the instance to run the experiment and shutdown / notify once finished.
    """
    imbalanced_benchmark.main()
    # when done, shutdown instance and notify
    instance_id = cfg['instance_id']
    requests.post(cfg['notification_url'], data={'value1': instance_id})
    stop_instance(instance_id)


def _load_dataset():
    dataset = cfg['dataset']
    user = cfg['remote']['user']
    dataset_url = cfg['dataset_urls'][dataset]
    urllib.request.urlretrieve(dataset_url, '/home/{}/datasets/dataset.tar.gz'.format(user))


# Command Line Interface
def main():
    instance_id = sys.argv[1]
    actions = sys.argv[2:]

    # actions called from workstation
    if 'start' in actions:
        start_instance(instance_id)
    if 'setup' in actions:
        setup_instance(instance_id)
    if 'dataset' in actions:
        dataset = actions[actions.index('dataset') + 1]
        _configure(instance_id, dataset)
        load_dataset(instance_id)
    if 'experiment' in actions:
        run_experiment(instance_id)
    if 'results' in actions:
        retrieve_results(instance_id)
    if 'stop' in actions:
        stop_instance(instance_id)
    if 'terminate' in actions:
        terminate_instance(instance_id)


    # actions called from remote
    if len(actions) < 1:
        action = instance_id
        if action == '_experiment':
            _run_experiment()
        if action == '_dataset':
            _load_dataset()


if __name__ == "__main__":
    main()
