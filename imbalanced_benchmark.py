import os
import warnings
import yaml
import pandas as pd
import re
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, average_precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import geometric_mean_score
from imbtools.evaluation import BinaryExperiment
from imbtools.evaluation import read_csv_dir

with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

def main():
    experiment_config = {
        'comment': 'Creditcard grid search',
        'experiment_repetitions': 5,
        'n_splits':10,
        'random_seed': int(os.urandom(1)[0] / 255 * (2**32)),
    }

    classifiers = [
        ('LR',LogisticRegression()),
        ('GB',GradientBoostingClassifier()),
        (
            'RF',RandomForestClassifier(),
            [{
                'criterion':['gini','entropy'],
                'n_estimators':[100]
            }]
        )
    ]
    oversampling_methods = [
        ('None',None),
        ('RandomOverSampler', RandomOverSampler()),
        ('SMOTE', SMOTE()),
        ('B1-SMOTE', SMOTE(kind='borderline1')),
        ('B2-SMOTE', SMOTE(kind='borderline2')),
        (
            'KMeansSMOTE', KMeansSMOTE(),
            [{
                'density_power': [None, 2],
                'smote_args': [{'k_neighbors': 5},{'k_neighbors': 100}],
                'kmeans_args': [
                    {'n_clusters':500, 'batch_size':1000, 'reassignment_ratio': 10**-5},
                    {'n_clusters':1000, 'batch_size':1000, 'reassignment_ratio': 10**-5},
                    {'n_clusters':1500, 'batch_size':1000, 'reassignment_ratio': 10**-5}
                ]
            }]
        )
    ]

    datasets = read_csv_dir(cfg['dataset_dir'])
    experiment = BinaryExperiment(
        datasets,
        classifiers,
        oversampling_methods,
        n_jobs=-1,
        experiment_repetitions=experiment_config['experiment_repetitions'],
        random_state=experiment_config['random_seed'],
        n_splits=experiment_config['n_splits'],
        scoring={
            'geometric_mean': make_scorer(geometric_mean_score),
            'average_precision': make_scorer(average_precision_score),
            'roc_auc': make_scorer(roc_auc_score),
            'f1': make_scorer(f1_score)
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Adapting smote_args\.k_neighbors')
        experiment.run()

    path = cfg['results_dir']
    if 'session_id' not in globals():
        session_id = (datetime.utcnow() + timedelta(hours=2,minutes=0)).strftime("%Y-%m-%d %Hh%M")

    os.makedirs('{}/{}'.format(path, session_id))

    experiment.save('{}/{}/experiment.p'.format(path, session_id))

    # stringify oversampling methods
    experiment_config['oversampling_methods'] = re.sub('\\n *',' ', str(oversampling_methods))
    # save experiment config
    pd.Series(experiment_config).to_csv('{}/{}/experiment_config.csv'.format(path, session_id))

if __name__ == "__main__":
    main()
