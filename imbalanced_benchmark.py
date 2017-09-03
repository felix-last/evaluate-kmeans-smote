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
        'comment': '...',
        'experiment_repetitions': 1,
        'n_splits':10,
        'random_seed': int(os.urandom(1)[0] / 255 * (2**32)),
    }

    classifiers = [
        ('LogisticRegression',LogisticRegression()),
        ('GradientBoosting',GradientBoostingClassifier()),
        (
            'RandomForest',RandomForestClassifier(),
            [{
                'criterion':['gini','entropy'],
                'n_estimators':[10,100]
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
                'minority_weight': [0.66, 1, 0.5],
                'density_power': [None, 2]
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
