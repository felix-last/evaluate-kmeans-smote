# %%
import imbtools.evaluation
import yaml
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from IPython.display import display
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette('muted')
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

path = cfg['results_dir']
session_id = '2017-10-09 08h24' #'2017-09-19 10h00'
try:
    df_optimal = pd.read_csv('{}/{}/optimal_stats.csv'.format(path, session_id), index_col=0)
    print('Experiment loaded from csv, computations skipped')
except:
    binary_experiment = imbtools.evaluation.load_experiment('{}/{}/experiment.p'.format(path, session_id))
    df_optimal = imbtools.evaluation.calculate_optimal_stats(binary_experiment)
    df_optimal.to_csv('{}/{}/optimal_stats.csv'.format(path, session_id))

#%%
metrics = ['f1'] # 'geometric_mean_score','average_precision',
classifiers = ['LogisticRegression'] #['KNN'] #,'GradientBoosting']
oversamplers = ['SMOTE', 'KMeansSMOTE'] # 'None','RandomOverSampler',
markers = ['|', '|']
datasets = df_optimal['Dataset'].unique()
#%%
fig, axes = plt.subplots(
    nrows=len(classifiers),
    ncols=len(metrics),
    figsize=(12,round(0.2*len(datasets))),
    sharex=True,
    sharey=True
)
if len(metrics) < 2:
    axes = [axes]
if len(classifiers) < 2:
    axes = [axes]
print(axes)
# sns.set_context('notebook')
for i,classifier in enumerate(classifiers):
    for j, metric in enumerate(metrics):
        ax = axes[i][j]
        df_optimal_filtered = df_optimal[
            (df_optimal['Metric'] == metric) &
            (df_optimal['Classifier'] == classifier) &
            (df_optimal['Oversampler'].isin(oversamplers))
        ]
        dataset_order = df_optimal_filtered[df_optimal_filtered['Oversampler'] == 'KMeansSMOTE'].sort_values(by='Mean CV score', ascending=False)['Dataset']
        df_optimal_filtered['Dataset'] = pd.Categorical(df_optimal_filtered['Dataset'], categories=dataset_order)
        df_optimal_filtered = df_optimal_filtered.sort_values(by='Dataset')

        for k, ovs in enumerate(df_optimal_filtered['Oversampler'].unique()):
            sns.stripplot(
                x='Mean CV score',
                y='Dataset',
                hue='Oversampler',
                marker=markers[k],
                data=df_optimal_filtered[df_optimal_filtered['Oversampler'] == ovs],
                order=dataset_order,
                size=6,
                linewidth=0.6,
                color=(0,0,0),
                ax=ax
            )
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set(xlim=(0,1))
        ax.legend().remove()

        # draw lines to left
        lines = ([
            [[0,x], [1,x]]
            for x, (_, group)
            in enumerate(df_optimal_filtered.groupby(['Dataset'])['Mean CV score'])
        ])
        colors = [(0.3,0.3,0.3),(0.7,0.7,0.7)]
        lc = mc.LineCollection(lines, colors=colors, linewidths=0.6, linestyles='dotted')
        ax.add_collection(lc)

        # draw line between points
        lines = ([[n,x] for n in group] for x, (_, group) in enumerate(df_optimal_filtered.groupby(['Dataset'])['Mean CV score']))
        lc = mc.LineCollection(lines, colors='red', linewidths=0.6)
        ax.add_collection(lc)

        # ax.set_markerfacecolor('none')

# create artists for legend
handles = [
    plt.Line2D((0,1),(0,0), marker=markers[i], linestyle='', label=ovs, color='black', markeredgewidth='0.5')
    for i,_ in enumerate(df_optimal_filtered['Oversampler'].unique())
]
labels = list(df_optimal_filtered['Oversampler'].unique())

# ax.legend()
# handles, labels = ax.get_legend_handles_labels()
# axes[0][0].legend(handles=handles, labels=labels,loc='lower left', numpoints=1)
for hax, classifier in zip(axes, classifiers):
    hax[0].set_ylabel(classifier)
for vax, metric in zip(axes[-1], metrics):
    vax.set_xlabel(metric)


#%%
# get metric + classifier combination with most average difference
df_optimal_filtered = df_optimal[(df_optimal['Oversampler'].isin(oversamplers)) & (df_optimal['Metric'].isin(metrics))]
smote = df_optimal_filtered[df_optimal_filtered['Oversampler'] == 'SMOTE']
kmsmote = df_optimal_filtered[df_optimal_filtered['Oversampler'] == 'KMeansSMOTE']
smote = smote.set_index(['Dataset','Classifier','Metric'])
kmsmote = kmsmote.set_index(['Dataset','Classifier','Metric'])
diff = (kmsmote['Mean CV score'] - smote['Mean CV score'])
classifier, metric = diff.groupby(level=['Classifier', 'Metric']).mean().sort_values(ascending=False).idxmax()


#%%
df_optimal_filtered[df_optimal_filtered['Oversampler'] == 'KMeansSMOTE'].sort_values(by='Mean CV score', ascending=False)

# %%
# dataset_order = df_optimal_filtered[df_optimal_filtered['Oversampler'] == 'KMeansSMOTE'].sort_values(by='Mean CV score', ascending=False)['Dataset']
df_optimal_filtered['Dataset'] = pd.Categorical(df_optimal_filtered['Dataset'], categories=dataset_order)
df_optimal_filtered.sort_values(by='Dataset')

#%%
dataset_order


##################################################################
# Determine correlation of dataset properties and gain
##################################################################
#%%
df_optimal_gain = df_optimal_filtered.drop('Std CV score', axis=1)
# df_optimal_gain = df_optimal_gain[
#     (df_optimal_gain['Metric'] == metric)
#     & (df_optimal_gain['Classifier'] == classifier)
# ]
df_optimal_gain = df_optimal_gain.set_index(['Dataset','Classifier','Oversampler','Metric'])
df_optimal_gain = df_optimal_gain.unstack('Oversampler')
df_optimal_gain.columns = df_optimal_gain.columns.droplevel(0)
df_optimal_gain.columns.name = None
df_optimal_gain['Gain'] = df_optimal_gain['KMeansSMOTE'] - df_optimal_gain['SMOTE']
df_optimal_gain = df_optimal_gain.drop(['KMeansSMOTE', 'SMOTE'], axis=1)
df_optimal_gain



#%%
datasets = imbtools.evaluation.read_csv_dir(cfg['dataset_dir'])
datasets = imbtools.evaluation.summarize_datasets(datasets)
datasets = datasets.set_index('Dataset name')
datasets.describe()

#%%
df_optimal_gain = df_optimal_gain.reset_index()
df_optimal_gain = pd.merge(
    left=df_optimal_gain,
    right=datasets,
    how='left',
    left_on='Dataset',
    right_index=True
)
df_optimal_gain = df_optimal_gain.set_index(['Dataset','Classifier','Metric'])
df_optimal_gain

#%%
fig, axes = plt.subplots(nrows=1,ncols=len(datasets.columns), figsize=(5*4,4), sharey=True)
for i, col in enumerate(datasets.columns):
    axes[i].set(ylim=(-0.01,df_optimal_gain['Gain'].max()+0.05))
    if col == 'Imbalance Ratio':
        axes[i].set(xlim=(0,datasets['Imbalance Ratio'].max()+5))
    sns.regplot(x=col,y='Gain', data= df_optimal_gain, ax=axes[i])



#%%
fig.savefig('{}/{}/gains_by_dataset_properties.png'.format(path, session_id))


##################################################################
# Find out how large gains are on average and how many times they are zero
##################################################################

#%%

#%%
df_optimal_gain_knn = df_optimal_gain.sort_index().loc[(slice(None),'KNN',slice(None)),:]
df_optimal_gain_knn.describe()

#%%
df_optimal_gain_knn[df_optimal_gain_knn['Gain'] == 0]

#%%
df_optimal_gain_knn[df_optimal_gain_knn['Gain'] > 0.1]


#%%
df_optimal_gain_knn.plot()


######################################################################
# Compare average gains between classifiers / metrics
######################################################################
#%%
df_optimal_filtered = df_optimal[
    (df_optimal['Metric'].isin(['average_precision','f1','geometric_mean_score']))
].set_index(['Dataset','Classifier','Metric'])
baseline = df_optimal_filtered[
    df_optimal_filtered['Oversampler'] == 'SMOTE'
].drop('Oversampler', axis=1)
successor = df_optimal_filtered[
    df_optimal_filtered['Oversampler'] == 'KMeansSMOTE'
].drop('Oversampler', axis=1)
df_gains = successor['Mean CV score'] - baseline['Mean CV score']
# make sure we are *always* at least as good
print((df_gains < 0).any())

#%%
# AVERAGE GAINS
df_avg_gains = df_gains.reset_index().groupby(['Classifier','Metric']).mean()
df_avg_gains = df_avg_gains.unstack()
df_avg_gains.columns = df_avg_gains.columns.droplevel()
sns.heatmap(
    data=df_avg_gains,
    annot=True,
    cmap=sns.color_palette('Blues')
)
plt.gcf().savefig('{}/{}/gains_versus_smote_avg.png'.format(path, session_id))

#%%
# MAX GAINS
df_max_gains = df_gains.reset_index().groupby(['Classifier', 'Metric']).max().drop('Dataset', axis=1)
df_max_gains = df_max_gains.unstack()
df_max_gains.columns = df_max_gains.columns.droplevel()
df_max_gains
sns.heatmap(
    data=df_max_gains,
    annot=True,
    cmap=sns.color_palette('Blues')
)
plt.gcf().savefig('{}/{}/gains_versus_smote_max.png'.format(path, session_id))


######################################################################
# Study noise generation by means of False Positives
######################################################################
# %%
df_optimal[
    (df_optimal['Oversampler'] == 'None')
    & (df_optimal['Metric'].isin(['tn','tp']))
]
# TP is constantly > TN, which doesnt make sense with no oversampling
# inverse them: fn -> fp, tn -> tp, ...
#%%
fp = 'fn'
fn = 'fp'
tp = 'tn'
tn = 'tp'

#%%
false_positives = df_optimal[
    (df_optimal['Metric'] == fp)
]
# false_positives = false_positives.drop(['Std CV score'], axis=1)
false_positives = false_positives.set_index(['Dataset', 'Classifier'])
baseline = false_positives[
    false_positives['Oversampler'] == 'SMOTE'
].drop('Oversampler', axis=1)
successor = false_positives[
    false_positives['Oversampler'] == 'KMeansSMOTE'
].drop('Oversampler', axis=1)
false_positives_reduction = baseline['Mean CV score'] - successor['Mean CV score']
relative_false_positives_reduction = (false_positives_reduction / baseline['Mean CV score']) * 100
datasets_ordered_by_mean_fp_reduction = relative_false_positives_reduction.unstack().mean(axis=1).sort_values(ascending=False).index
relative_false_positives_reduction = relative_false_positives_reduction.unstack()
relative_false_positives_reduction = relative_false_positives_reduction.loc[
    datasets_ordered_by_mean_fp_reduction, :].stack()

#%%
fig, ax = plt.subplots(1,1,figsize=(20,2))
sns.heatmap(
    data=relative_false_positives_reduction.unstack(0),
    cmap=sns.color_palette('Blues'),
    ax=ax,
    cbar_kws={'label': '% of False Positives Reduction'}
)
plt.gcf().savefig('{}/{}/false_positives_reduction_vs_smote.png'.format(path, session_id),bbox_inches='tight')

#%%
# AVERAGE ACROSS DATASETS
avg_relative_false_positives_reduction = relative_false_positives_reduction.reset_index(
).groupby('Classifier').mean()
avg_relative_false_positives_reduction

# %%
# check that we are not just doing random oversampling to minimize FP
baseline = false_positives[
    false_positives['Oversampler'] == 'RandomOverSampler'
].drop('Oversampler', axis=1)
false_positives_reduction = baseline['Mean CV score'] - \
    successor['Mean CV score']
relative_false_positives_reduction = (
    false_positives_reduction / baseline['Mean CV score']) * 100
relative_false_positives_reduction
fig, ax = plt.subplots(1, 1, figsize=(20, 2))
sns.heatmap(
    data=false_positives_reduction.unstack(0),
    cmap=sns.color_palette('RdBu',10),
    ax=ax,
    cbar_kws={'label': 'Absolute False Positives Reduction'}
)
