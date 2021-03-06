import os
import warnings
import yaml
import pandas as pd
import re
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from kmeans_smote import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import geometric_mean_score
from imbtools.evaluation import BinaryExperiment
from imbtools.evaluation import read_csv_dir

with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

def main():
    experiment_config = {
        'comment': 'Keel run',
        'experiment_repetitions': 5,
        'n_splits':5,
        'random_seed': int(os.urandom(1)[0] / 255 * (2**32)),
    }

    classifiers = [
        (
            'LR', LogisticRegression()
        ),(
            'GBM',GradientBoostingClassifier(),
            [{
                'n_estimators': [50, 100, 200]
            }]
        ),(
            'KNN',KNeighborsClassifier(),
            [{
                'n_neighbors': [3,5,8]
            }]
        )
    ]
    oversampling_methods = [
        ('None',None),
        ('RandomOverSampler', RandomOverSampler()),
        (
            'SMOTE', SMOTE(),
            [{
                'k_neighbors': [3,5,20]
            }]
        ),
        (
            'B1-SMOTE', SMOTE(kind='borderline1'),
            [{
                'k_neighbors': [3,5,20]
            }]
        ),
        (
            'B2-SMOTE', SMOTE(kind='borderline2'),
            [{
                'k_neighbors': [3,5,20]
            }]
        ),
        (
            'KMeansSMOTE', KMeansSMOTE(),
            [
                {
                    'imbalance_ratio_threshold': [1,float('Inf')],
                    'density_power': [0, 2, None], # None corresponds to n_features
                    'smote_args': [
                        {'k_neighbors': 3},{'k_neighbors': 5},
                        {'k_neighbors': 20},{'k_neighbors': float('Inf')}
                    ],
                    'kmeans_args': [
                        {'n_clusters': 2}, {'n_clusters': 20}, {'n_clusters': 50},
                        {'n_clusters': 100}, {'n_clusters':250}, {'n_clusters':500}
                    ],
                    'use_minibatch_kmeans':[True],
                    'n_jobs':[-1]
                },
                # SMOTE Limit Case
                {
                    'imbalance_ratio_threshold': [float('Inf')],
                    'kmeans_args': [{'n_clusters':1}],
                    'smote_args': [
                        {'k_neighbors': 3},{'k_neighbors': 5}
                    ],
                    'use_minibatch_kmeans':[True],
                    'n_jobs':[-1]
                }
            ]
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
        scoring=['geometric_mean_score', 'average_precision','roc_auc','f1', 'fp', 'fn', 'tp', 'tn']
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
