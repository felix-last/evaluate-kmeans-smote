# coding: utf-8

# <markdowncell>
# # 1 Introduction
# This notebook tests the effectiveness of K-Means SMOTE (unpublished).
# # 2 Imports

# <codecell>
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE
from imbtools.evaluation import BinaryExperiment
from plot_imbalanced_benchmark import createPdf
import warnings
import yaml
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# <markdowncell>
# # 3 Experiment
# #### 3.1 Configure experiment

# <codecell>
experiment_config = {
    'comment': '...',
    'experiment_repetitions': 10,
    'n_splits':3,
    'random_seed': int(os.urandom(1)[0] / 255 * (2**32)),
}

# <codecell>
classifiers = [LogisticRegression(), GradientBoostingClassifier()]
oversampling_methods = [
    None,
    RandomOverSampler(),
    SMOTE(),
    SMOTE(kind='borderline1'),
    KMeansSMOTE()
]

# <codecell>
experiment = BinaryExperiment(
    cfg['dataset_dir'],
    classifiers,
    oversampling_methods,
    n_jobs=1,
    experiment_repetitions=experiment_config['experiment_repetitions'],
    random_state=experiment_config['random_seed'],
    n_splits=experiment_config['n_splits']
)

# <markdowncell>
# #### 3.2 Run experiment
# <codecell>
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', message='Adapting smote_args\.k_neighbors')
    experiment.run(logging_results=False)

# <markdowncell>
# #### 3.3 Datasets summary

# <codecell>
experiment.datasets_summary_

# <markdowncell>
# #### 3.4 Mean CV results

# <codecell>
experiment.mean_cv_results_


# <codecell>
experiment.mean_ranking_results_


# #### 3.5 Standard deviation CV results

# <codecell>
experiment.std_cv_results_

# <markdowncell>
# #### 3.6. Oversampling methods mean ranking

# <codecell>
experiment.mean_ranking_results_

# <markdowncell>
# #### 3.7 Friedman test

# <codecell>
experiment.friedman_test_results_


# <codecell>
experiment.mean_cv_results_
experiment.std_cv_results_
experiment.friedman_test_results_


# <codecell>
experiment.mean_cv_results_[(experiment.mean_cv_results_['Dataset']=='libra') & ((experiment.mean_cv_results_['Oversampling method'] =='KMeansSMOTE') | (experiment.mean_cv_results_['Oversampling method'] == 'SMOTE'))]


# <codecell>
experiment.mean_cv_results_[(experiment.mean_cv_results_['Dataset']=='segment') & ((experiment.mean_cv_results_['Oversampling method'] =='KMeansSMOTE') | (experiment.mean_cv_results_['Oversampling method'] == 'SMOTE'))]

# <markdowncell>
# ## Save results to file

# <codecell>
from datetime import datetime, timedelta
import pandas as pd
import re
path = cfg['results_dir']
if 'session_id' not in globals():
    session_id = (datetime.utcnow() + timedelta(hours=2,minutes=0)).strftime("%Y-%m-%d %Hh%M")

os.makedirs('{}/{}'.format(path, session_id))


# <codecell>
experiment.datasets_summary_.to_csv('{}/{}/datasets_summary.csv'.format(path, session_id))
experiment.friedman_test_results_.to_csv('{}/{}/friedman_test_results.csv'.format(path, session_id))
experiment.mean_cv_results_.to_csv('{}/{}/mean_cv_results.csv'.format(path, session_id))
experiment.mean_ranking_results_.to_csv('{}/{}/mean_ranking_results.csv'.format(path, session_id))
experiment.std_cv_results_.to_csv('{}/{}/std_cv_results.csv'.format(path, session_id))

# <codecell>
re.sub('\\n *',' ', str(oversampling_methods))

# <codecell>
# stringify oversampling methods
experiment_config['oversampling_methods'] = re.sub('\\n *',' ', str(oversampling_methods))
# save experiment config
pd.Series(experiment_config).to_csv('{}/{}/experiment_config.csv'.format(path, session_id))

# <codecell>
createPdf(session_id)
