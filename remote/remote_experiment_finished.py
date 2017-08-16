import requests
import yaml
import remote

with open("../config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

def main():
    instance_id = cfg['instance_id']
    r = requests.post(cfg['notification_url'], data = {'value1':instance_id})
    remote.stop_instance(instance_id)

if __name__ == "__main__":
    main()
