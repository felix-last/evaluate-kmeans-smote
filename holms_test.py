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
session_id = 'current'
friedman_test_results = pd.read_csv('{}/{}/friedman_test_results.csv'.format(path, session_id), index_col=0)

# <codecell>
p_values = friedman_test_results['p-value']
p_values
multipletests(p_values, alpha=0.05, method='h')
