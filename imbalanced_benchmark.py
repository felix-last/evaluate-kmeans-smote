import os
import warnings
import yaml
import pandas as pd
import re
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imbtools.evaluation import BinaryExperiment

with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

def main():
    experiment_config = {
        'comment': '100 repetitions of classic datasets',
        'experiment_repetitions': 100,
        'n_splits':3,
        'random_seed': int(os.urandom(1)[0] / 255 * (2**32)),
    }

    classifiers = [LogisticRegression(), GradientBoostingClassifier()]
    oversampling_methods = [
        None,
        RandomUnderSampler(),
        RandomOverSampler(),
        SMOTE(),
        SMOTE(kind='borderline1'),
        KMeansSMOTE() #kmeans_args={'n_clusters': 1000, 'batch_size':1000, 'reassignment_ratio': 10**-4}
    ]

    experiment = BinaryExperiment(
        cfg['dataset_dir'],
        classifiers,
        oversampling_methods,
        n_jobs=-1,
        experiment_repetitions=experiment_config['experiment_repetitions'],
        random_state=experiment_config['random_seed'],
        n_splits=experiment_config['n_splits']
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Adapting smote_args\.k_neighbors')
        experiment.run(logging_results=False)

    path = cfg['results_dir']
    if 'session_id' not in globals():
        session_id = (datetime.utcnow() + timedelta(hours=2,minutes=0)).strftime("%Y-%m-%d %Hh%M")

    os.makedirs('{}/{}'.format(path, session_id))

    try:
        experiment.datasets_summary_.to_csv('{}/{}/datasets_summary.csv'.format(path, session_id))
        experiment.friedman_test_results_.to_csv('{}/{}/friedman_test_results.csv'.format(path, session_id))
        experiment.mean_cv_results_.to_csv('{}/{}/mean_cv_results.csv'.format(path, session_id))
        experiment.mean_ranking_results_.to_csv('{}/{}/mean_ranking_results.csv'.format(path, session_id))
        experiment.std_cv_results_.to_csv('{}/{}/std_cv_results.csv'.format(path, session_id))
    except: pass
    try:
        experiment.roc_.to_csv('{}/{}/roc.csv'.format(path, session_id))
    except: pass

    # stringify oversampling methods
    experiment_config['oversampling_methods'] = re.sub('\\n *',' ', str(oversampling_methods))
    # save experiment config
    pd.Series(experiment_config).to_csv('{}/{}/experiment_config.csv'.format(path, session_id))

if __name__ == "__main__":
    main()
