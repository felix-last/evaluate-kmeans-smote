import yaml

def main():
    with open("../config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    cfg['dataset_dir'] = '~/datasets'
    cfg['results_dir'] = '~/results'

    with open("../config.yml", 'w') as ymlfile:
        yaml.dump(cfg, ymlfile, default_flow_style=False)

if __name__ == "__main__":
    main()
