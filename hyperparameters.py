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
# get optimal stats
optimal = imbtools.evaluation.calculate_optimal_stats(binary_experiment, return_optimal_params=True)

#%%
# split oversampler and classifier column into to columns
optimal['ClassifierKind'] = optimal['Classifier'].apply(lambda o: o[0])
optimal['OversamplerKind'] = optimal['Oversampler'].apply(lambda o: o[0])
optimal['ClassifierConfiguration'] = optimal['Classifier'].apply(lambda o: o[1])
optimal['OversamplerConfiguration'] = optimal['Oversampler'].apply(lambda o: o[1])
# filter only kmeans smote rows
optimal = optimal[ optimal['OversamplerKind'] == 'KMeansSMOTE' ]
# Remove unnecessary metrics
optimal = optimal[ ~optimal['Metric'].isin(['tp','tn','fp','fn']) ]
optimal

#%%
# create a well-readable table
table_optimal = optimal.drop(['Oversampler','Classifier', 'OversamplerKind', 'OversamplerConfiguration', 'ClassifierConfiguration', 'Mean CV score', 'Std CV score'], axis=1)
# add configuration of kmeans smote as columns
table_optimal['Score'] = optimal['Mean CV score']
table_optimal['SD'] = optimal['Std CV score']
table_optimal['k'] = optimal['OversamplerConfiguration'].apply(lambda o: o['kmeans_args']['n_clusters'])
table_optimal['knn'] = optimal['OversamplerConfiguration'].apply(lambda o: o['smote_args']['k_neighbors'])
table_optimal['de'] = optimal['OversamplerConfiguration'].apply(lambda o: o['density_power'])
table_optimal['irt'] = optimal['OversamplerConfiguration'].apply(lambda o: o['imbalance_ratio_threshold'])
# add classifier configurations as columns
# since we only have max 1 variable parameter per classifier, use only one column
def get_classifier_param(o):
    if 'n_estimators' in o:
        return o['n_estimators']
    elif 'n_neighbors' in o:
        return o['n_neighbors']
    else:
        return np.nan
table_optimal['Clf'] = optimal['ClassifierConfiguration'].apply(get_classifier_param)
# empty classifier parameter when it's not applicable
table_optimal['Clf'] = table_optimal['Clf'].replace(np.nan, '')
# empty density exponent when it's not applicable
table_optimal['de'] = table_optimal['de'].replace(np.nan, '')
table_optimal = table_optimal.set_index(['Dataset','ClassifierKind','Metric'])
table_optimal = table_optimal.unstack('Metric')
table_optimal = table_optimal.swaplevel(0,1, axis=1)
table_optimal.sortlevel(0, axis=1, inplace=True)
# replace infs with string to allow rounding
table_optimal = table_optimal.replace([np.inf, -np.inf], '∞')
table_optimal = table_optimal.round(2)
# choose only one metric, otherwise they are too many
table_optimal = table_optimal['average_precision']
table_optimal
#%%
# export
print(table_optimal.to_latex())

# %%
# get table of datasets
datasets = imbtools.evaluation.read_csv_dir(cfg['dataset_dir'])
print(imbtools.evaluation.summarize_datasets(datasets).to_latex())

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
