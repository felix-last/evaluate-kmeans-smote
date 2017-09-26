#%%
import imbtools.evaluation
import yaml
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from IPython.display import display
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#%%
path = cfg['results_dir']
session_id = '2017-09-19 10h00'
binary_experiment = imbtools.evaluation.load_experiment('{}/{}/experiment.p'.format(path, session_id))
res = binary_experiment.results_
display(res)

# %%
# Average the oversamplers with the same name - they are there once for each run
res = res.groupby(['Dataset','Classifier','Oversampler','Metric'],as_index=False).mean()

# Remove other oversamplers
number_regex = re.compile(r'[0-9]+')
res['OversamplerKind'] = res['Oversampler'].replace(number_regex,'')
res = res[ res['OversamplerKind'] == 'KMeansSMOTE' ]

# Remove unnecessary metrics
res = res[ ~res['Metric'].isin(['tp','tn','fp','fn']) ]
# Add a column to group classifiers of same kind (ignoring hyperparams)

res['ClassifierKind'] = res['Classifier'].replace(number_regex,'')
res.drop('OversamplerKind', axis=1, inplace=True)
display(res)

#%%
# find best
best_idx = res.groupby(['Dataset', 'Metric', 'ClassifierKind'])['CV score'].idxmax()
best = res.loc[best_idx.values]
best = best.set_index(['Dataset','ClassifierKind','Metric'])
best = best.unstack('Metric')
best = best.swaplevel(0,1, axis=1)
best.sortlevel(0, axis=1, inplace=True)
# best.loc['iris']
best
# display(res[(res['Dataset'] == 'iris') & (res['Oversampler'] == 'KMeansSMOTE99')])

#%%
# Look at one metric and plot the various oversamplers
pr_res = res[res['Metric'] == 'average_precision']
# look at only one classifier
pr_res = pr_res[pr_res['ClassifierKind'] == 'LogisticRegression']
row_count = np.unique(pr_res['Dataset']).size
fig,axes = plt.subplots(
    row_count,1,
    figsize=(10,row_count * 10)
)
for i, dataset in enumerate(np.unique(pr_res['Dataset'])):
    pr_res_dataset = pr_res[pr_res['Dataset'] == dataset]
    oversamplers = pr_res_dataset['Oversampler'].reset_index(drop=True)
    oversamplers.index = oversamplers.index + 1
    ax = axes[i]
    ax.plot(oversamplers.index, pr_res_dataset['CV score'].values)
    ax.set_title('AUC PR for Logistic Regression on {}'.format(dataset))
fig.savefig('result.pdf')

#%%
pd.DataFrame(binary_experiment.oversamplers_)

############# GOAL ################################################
#                               F1-Score            G-Mean
# iris
#   Logistic Regression         IRT = 1             IRT = 2
#                               n_clusters = 50     n_clusters = 50
#   Random Forest               IRT = 1             IRT = 2
#                               n_clusters = 50     n_clusters = 50
# wine
#   Logistic Regression         IRT = 1             IRT = 2
#                               n_clusters = 50     n_clusters = 50
#   Random Forest               IRT = 1             IRT = 2
#                               n_clusters = 50     n_clusters = 50
