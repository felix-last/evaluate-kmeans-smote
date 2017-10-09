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
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

path = cfg['results_dir']
session_id = '2017-09-19 10h00'
binary_experiment = imbtools.evaluation.load_experiment('{}/{}/experiment.p'.format(path, session_id))
df_optimal = imbtools.evaluation.calculate_optimal_stats(binary_experiment)

#%%
metrics = ['f1'] # 'geometric_mean_score','average_precision',
classifiers = ['LogisticRegression'] #['KNN'] #,'GradientBoosting']
oversamplers = ['SMOTE', 'KMeansSMOTE'] # 'None','RandomOverSampler',
markers = ['|','|',(6,2,60), '.','|', '_',]

fig, axes = plt.subplots(
    nrows=len(classifiers),
    ncols=len(metrics),
    figsize=(12,12),
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
        for k, ovs in enumerate(df_optimal_filtered['Oversampler'].unique()):
            sns.stripplot(
                x='Mean CV score',
                y='Dataset',
                hue='Oversampler',
                marker=markers[k],
                data=df_optimal_filtered[df_optimal_filtered['Oversampler'] == ovs],
                order=np.sort(df_optimal_filtered['Dataset'].unique()),
                size=4,
                linewidth=0.5,
                color='black',
                ax=ax
            )
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set(xlim=(0,1))
        ax.legend().remove()

        # draw line between points
        lines = ([[n,x] for n in group] for x, (_, group) in enumerate(df_optimal_filtered.groupby(['Dataset'])['Mean CV score']))
        lc = mc.LineCollection(lines, colors='red', linewidths=0.8)
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
diff.groupby(level=['Classifier', 'Metric']).mean().sort_values(ascending=False).plot.bar()
