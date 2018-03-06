# <codecell>
import imbtools.evaluation
from imbtools.evaluation import calculate_friedman_test
import yaml
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline
from statsmodels.stats.multitest import multipletests
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
# <codecell>
path = cfg['results_dir']
session_id = '2017-10-09 08h24'
try:
    friedman_test_results = pd.read_csv(
        '{}/{}/friedman_test_results.csv'.format(path, session_id), index_col=0)
    print('Friedman test loaded from csv, computations skipped')
except:
    friedman_test_results = calculate_friedman_test(
        binary_experiment)
    friedman_test_results.to_csv(
        '{}/{}/friedman_test_results.csv'.format(path, session_id))

# <codecell>
metrics = ['average_precision','geometric_mean_score','f1']
metric_names = {
    'geometric_mean_score': 'g-mean',
    'average_precision': 'AUPRC',
    'f1': 'F1',
    'roc_auc_score': 'ROC AUC'
}

# <codecell>
p_values = friedman_test_results['p-value']
p_values

# <codecell>
significance, adjusted_p, _, _ = multipletests(p_values, alpha=0.05, method='h')
adjusted_p
holms_test_results = friedman_test_results[['Classifier','Metric']]
holms_test_results['Adjusted p'] = adjusted_p
holms_test_results['Significance'] = significance

# <codecell>
# compute significance in asterisk-notation
def asterisks_from_pval(p):
    significance = 'ns'
    for pv, sym in [(0.05, '*'), (0.01, '**'), (0.001, '***'), (0.0001, '****')]:
        if p <= pv:
            significance = sym
    return significance
holms_test_results['Significance Level'] = holms_test_results['Adjusted p'].apply(asterisks_from_pval)
holms_test_results

# <codecell>
# Filter only the relevant metrics
holms_test_results = holms_test_results[holms_test_results['Metric'].isin(metrics)]

# <codecell>
# save results to file
holms_test_results.to_csv(
    '{}/{}/holms_test_results.csv'.format(path, session_id), index_col=0)
