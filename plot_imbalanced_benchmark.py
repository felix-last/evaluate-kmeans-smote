# coding: utf-8

# <codecell>
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import Counter
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('deep')
sns.set_color_codes(palette="deep")
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import imbtools.evaluation
from imbtools.evaluation import calculate_stats, calculate_optimal_stats, calculate_optimal_stats_wide, calculate_ranking, calculate_mean_ranking, calculate_friedman_test
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

order = ['KMeansSMOTE','SMOTE','B1-SMOTE','B2-SMOTE','RandomOverSampler','None']

metric_names = {
    'geometric_mean_score':'g-mean',
    'average_precision':'AUPRC',
    'f1':'F1',
    'roc_auc_score': 'ROC AUC'
}

def create_pdf(session_id, ranking=True, mean_cv=True, comparison=True, roc=True, metrics='all', verbose=True):
    path = cfg['results_dir']

    experiment = load_experiment(session_id, ranking, mean_cv, comparison, roc, verbose)

    with PdfPages('{0}/{1}/{1}.pdf'.format(path, session_id)) as pdf:
        if ranking and 'friedman_test_results' in experiment:
            if verbose: print('Plotting ranking ...')
            fig, title = plot_mean_ranking(experiment['mean_ranking_results'], experiment['friedman_test_results'], metrics=metrics)
            pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")
        if mean_cv and 'mean_cv_results' in experiment:
            if verbose: print('Plotting mean cv results ...')
            fig, title = plot_cross_validation_mean_results(experiment['mean_cv_results'], experiment['std_cv_results'])
            pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")
        if comparison and 'mean_cv_results' in experiment:
            if verbose: print('Plotting comparison ...')
            fig = plot_comparison(experiment['mean_cv_results'], metrics)
            pdf.savefig(fig)
        if roc and 'roc' in experiment:
            if verbose: print('Plotting roc ...')
            figs, titles = plot_roc(experiment)
            for fig, title in zip(figs, titles):
                pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")

def create_png(session_id, verbose=True, ranking=True, mean_cv=True, comparison=True, roc=True, metrics='all'):
    path = cfg['results_dir']

    experiment = load_experiment(session_id, ranking=ranking, mean_cv=mean_cv, comparison=comparison, roc=roc)

    if ranking and 'friedman_test_results' in experiment:
        if verbose: print('Plotting ranking ...')
        fig, title = plot_mean_ranking(experiment['mean_ranking_results'], experiment['friedman_test_results'], metrics=metrics)
        fig.savefig('{0}/{1}/ranking.png'.format(path, session_id), dpi=300)
    if mean_cv and 'mean_cv_results' in experiment:
        if verbose: print('Plotting mean cv results ...')
        fig, title = plot_cross_validation_mean_results(experiment['mean_cv_results'], experiment['std_cv_results'])
        fig.savefig('{0}/{1}/mean.png'.format(path, session_id), dpi=300)
    if comparison and 'mean_cv_results' in experiment:
        if verbose: print('Plotting comparison ...')
        fig = plot_comparison(experiment['mean_cv_results'], metrics)
        fig.savefig('{0}/{1}/comparison.png'.format(path, session_id), dpi=300)
    if roc and 'roc' in experiment:
        if verbose: print('Plotting roc ...')
        figs, titles = plot_roc(experiment)
        for i, (fig, title) in enumerate(zip(figs, titles)):
            fig.savefig('{0}/{1}/roc{2}.png'.format(path, session_id, i),
                        bbox_extra_artists=(title,), bbox_inches="tight", dpi=300)

def load_experiment(session_id, ranking=True, mean_cv=True, comparison=True, roc=True, verbose=True):
    path = cfg['results_dir']
    if verbose: print('Loading experiment from disk ...')
    if 'experiment.p' in os.listdir( os.path.join(path, session_id) ):
        # new imbtools 1.0.0 format
        binary_experiment = imbtools.evaluation.load_experiment('{}/{}/experiment.p'.format(path, session_id))
        if verbose: print('Computing stats ...')
        experiment = {}
        if ranking:
            try:
                mean_ranking_results = pd.read_csv('{}/{}/mean_ranking_results.csv'.format(path, session_id), index_col=0)
                friedman_test_results = pd.read_csv('{}/{}/friedman_test_results.csv'.format(path, session_id), index_col=0)
                if verbose: print('Mean ranking loaded from csv, computations skipped')
            except:
                mean_ranking_results = calculate_mean_ranking(binary_experiment)
                friedman_test_results = calculate_friedman_test(binary_experiment)
                mean_ranking_results.to_csv('{}/{}/mean_ranking_results.csv'.format(path, session_id))
                friedman_test_results.to_csv('{}/{}/friedman_test_results.csv'.format(path, session_id))
            experiment['mean_ranking_results'] = mean_ranking_results
            experiment['friedman_test_results'] = friedman_test_results
        if (mean_cv | comparison):
            try:
                optimal_stats = pd.read_csv('{}/{}/optimal_stats.csv'.format(path, session_id), index_col=0)
                if verbose: print('Optimal stats loaded from csv, computations skipped')
            except:
                optimal_stats = calculate_optimal_stats(binary_experiment)
                optimal_stats.to_csv('{}/{}/optimal_stats.csv'.format(path, session_id))
            experiment['mean_cv_results'] = optimal_stats.drop('Std CV score', axis=1)
            experiment['std_cv_results'] = optimal_stats.drop('Mean CV score', axis=1)
    else:
        # old format
        experiment = {}
        try:
            experiment['datasets_summary'] = pd.read_csv(
                '{}/{}/datasets_summary.csv'.format(path, session_id))
            experiment['friedman_test_results'] = pd.read_csv(
                '{}/{}/friedman_test_results.csv'.format(path, session_id))
            experiment['mean_cv_results'] = pd.read_csv(
                    '{}/{}/mean_cv_results.csv'.format(path, session_id))
            experiment['mean_cv_results'].drop('Unnamed: 0', inplace=True)
            experiment['mean_ranking_results'] = pd.read_csv(
                '{}/{}/mean_ranking_results.csv'.format(path, session_id))
            experiment['std_cv_results'] = pd.read_csv(
                '{}/{}/std_cv_results.csv'.format(path, session_id))
            experiment['std_cv_results'].drop('Unnamed: 0', inplace=True)
        except: pass
        try:
            experiment['roc'] = pd.read_csv(
                '{}/{}/roc.csv'.format(path, session_id))
        except: pass
    if verbose: print('Experiment loaded.')
    return experiment

def plot_mean_ranking(mean_ranking_results, friedman_test_results, metrics='all', make_title=True):
    if metrics is 'all':
        metrics = np.unique(mean_ranking_results['Metric'])
    classifiers = np.unique(mean_ranking_results['Classifier'])

    # change order
    desired_order = ['Classifier','Metric'] + order
    if len(mean_ranking_results.columns) == len(desired_order):
        mean_ranking_results = mean_ranking_results.loc[:,desired_order]

    row_count = len(classifiers)
    col_count = len(metrics)
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(
        3 * col_count, 4 * row_count), sharey='row', sharex='col')
    if row_count == 1:
        axes = [axes]

    for j, metric in enumerate(metrics):
        metric_name = metric_names[metric] if metric in metric_names else metric
        axes[0][j].set_title(metric_name)

    for i, classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            ranking_filtered = mean_ranking_results
            ranking_filtered = ranking_filtered[
                (ranking_filtered['Classifier'] == classifier)
                & (ranking_filtered['Metric'] == metric)
            ]
            ranking_filtered = ranking_filtered.drop(
                ['Classifier', 'Metric'], axis=1)
            # oversampling_method_column = 'Oversampling method' if 'Oversampling method' in ranking_filtered.columns else 'Oversampler'
            # ranking_filtered[oversampling_method_column].replace({'None': 0,'RandomOverSampler':1,'SMOTE':2, 'B2-SMOTE':3, 'B2-SMOTE':4, 'KMeansSMOTE':100})
            # ranking_filtered.sort_values(oversampling_method_column, axis=1)
            ax = axes[i][j]
            methods_encoded = np.asarray(
                [i for i, _ in enumerate(ranking_filtered.columns)])
    #         method_colors = sns.husl_palette(n_colors=len(methods_encoded))

            # invert the ranking so the highest bar is the best
            worst_rank = math.ceil(ranking_filtered.max().max())
            ranking_filtered = (ranking_filtered - worst_rank) * -1
            ticks = list(range(1,worst_rank))
            ax.set_yticks(ticks)
            ax.set_yticklabels(reversed(ticks))
            ax.set_ylim(0,worst_rank-1)
            ax.bar(
                methods_encoded,
                np.asarray(ranking_filtered).transpose(),
                0.5,
                #             color=method_colors
            )
            ax.set_xticks(methods_encoded + 0.25)
            ax.set_xticklabels(ranking_filtered.columns, rotation='vertical')
            if j is 0:
                ax.set_ylabel('{}\n\n{}'.format(classifier,'Mean Rank'))

            ax.grid(axis='x', which='both', linewidth=0)

            corresponding_friedman = np.asarray(friedman_test_results[
                (friedman_test_results
                 ['Classifier'] == classifier)
                & (friedman_test_results['Metric'] == metric)
            ]['p-value'])[0]
            significance = 'ns'
            for pv,sym in [(0.05, '*'), (0.01, '**'), (0.001, '***'), (0.0001, '****')]:
                if corresponding_friedman <= pv:
                    significance = sym
            ax.text(.98, .98, significance, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    fig.tight_layout()
    title = plt.suptitle(
        'Mean Ranking Results + Friedman Test', fontsize=14, y=1.02)
    if make_title:
        return fig, title
    else:
        return fig


def plot_cross_validation_mean_results(mean_cv_results, std_cv_results=None):
    datasets = np.unique(mean_cv_results['Dataset'])
    classifiers = np.unique(mean_cv_results['Classifier'])
    metrics = np.unique(mean_cv_results['Metric'])

    row_count = len(classifiers) * len(datasets)
    col_count = len(metrics)
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(
        3 * col_count, 5 * row_count), sharey='row')
    if row_count == 1:
        axes = [axes]
    for j, metric in enumerate(metrics):
        metric_name = metric_names[metric] if metric in metric_names else metric
        axes[0][j].set_title(metric_name)
    for i, classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            for k, dataset in enumerate(datasets):
                mean_filtered = mean_cv_results
                mean_filtered = mean_filtered[
                    (mean_filtered['Classifier'] == classifier)
                    & (mean_filtered['Metric'] == metric)
                    & (mean_filtered['Dataset'] == dataset)
                ]
                mean_filtered = mean_filtered.drop(
                    ['Classifier', 'Metric', 'Dataset'], axis=1)

                if std_cv_results is not None:
                    std_filtered = std_cv_results
                    std_filtered = std_filtered[
                        (std_filtered['Classifier'] == classifier)
                        & (std_filtered['Metric'] == metric)
                        & (std_filtered['Dataset'] == dataset)
                    ]
                    std_filtered = std_filtered.drop(
                        ['Classifier', 'Metric', 'Dataset'], axis=1)
                    yerr = std_filtered['Std CV score']
                else:
                    yerr = None

                ax = axes[len(datasets) * i + k][j]
                oversampling_method_column = 'Oversampling method' if 'Oversampling method' in mean_filtered.columns else 'Oversampler'
                methods_encoded = np.asarray(
                    [i for i, _ in enumerate(mean_filtered[oversampling_method_column])])
    #             method_colors = sns.husl_palette(n_colors=len(methods_encoded))
                bars = ax.bar(
                    methods_encoded,
                    mean_filtered['Mean CV score'],
                    0.5,
                    #                 color= method_colors,
                    yerr=yerr,
                    ecolor='black'
                )
                if(max(mean_filtered['Mean CV score']) <= 1):
                    ax.set_ylim((0, 1))

                ax.set_xticks(methods_encoded + 0.25)
                ax.set_xticklabels(
                    mean_filtered[oversampling_method_column], rotation='vertical')

                if j is 0:
                    ax.set_ylabel('{} - {}'.format(classifier, dataset))
    # axes[0][0].legend(bars, np.asarray(mean_filtered[oversampling_method_column]))
    fig.tight_layout()
    title = plt.suptitle(
        'Mean Cross-Validation Result + Standard Deviation', fontsize=14,  y=1.02)
    return fig, title

def plot_comparison(mean_cv_results, metrics='all'):
    df_optimal = mean_cv_results
    oversamplers = ('SMOTE', 'KMeansSMOTE')
    if metrics is 'all':
        metrics = np.unique(df_optimal['Metric'])

    # get metric + classifier combination with most average difference
    df_optimal_filtered = df_optimal[(df_optimal['Oversampler'].isin(oversamplers)) & (df_optimal['Metric'].isin(metrics))]
    base = df_optimal_filtered[df_optimal_filtered['Oversampler'] == oversamplers[0]]
    successor = df_optimal_filtered[df_optimal_filtered['Oversampler'] == oversamplers[1]]
    base = base.set_index(['Dataset','Classifier','Metric'])
    successor = successor.set_index(['Dataset','Classifier','Metric'])
    diff = (successor['Mean CV score'] - base['Mean CV score'])
    classifier, metric = diff.groupby(level=['Classifier', 'Metric']).mean().sort_values(ascending=False).idxmax()

    metrics = [metric]
    classifiers = [classifier]

    markers = ['|','|']

    fig, axes = plt.subplots(
        nrows=len(classifiers),
        ncols=len(metrics),
        sharex=True,
        figsize=(12,round(0.2*len(df_optimal['Dataset'].unique()))),
        sharey=True
    )
    if len(metrics) < 2:
        axes = [axes]
    if len(classifiers) < 2:
        axes = [axes]
    for i,classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            df_optimal_filtered = df_optimal[
                (df_optimal['Metric'] == metric) &
                (df_optimal['Classifier'] == classifier) &
                (df_optimal['Oversampler'].isin(oversamplers))
            ]

            dataset_order = df_optimal_filtered[df_optimal_filtered['Oversampler'] == oversamplers[1]].sort_values(by='Mean CV score', ascending=False)['Dataset']
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
                    size=10,
                    alpha=1,
                    linewidth=0.5,
                    color=(0,0,0),
                    ax=ax
                )
            ax.set_ylabel('')
            ax.set_xlabel('')

            ax.set(xlim=(0,1))
            ax.legend().remove()

            # draw lines to left
            colors = [(0.3,0.3,0.3),(0.7,0.7,0.7)]
            lines = ([
            [[0,x], [1,x]]
                for x, (_, group)
                in enumerate(df_optimal_filtered.groupby(['Dataset'])['Mean CV score'])
            ])
            lc = mc.LineCollection(lines, colors=colors, linewidths=0.8, linestyles='dotted')
            ax.add_collection(lc)

            # draw line between points
            lines = ([[n,x] for n in group] for x, (_, group) in enumerate(df_optimal_filtered.groupby(['Dataset'])['Mean CV score']))
            lc = mc.LineCollection(lines, colors='red', linewidths=0.8)
            ax.add_collection(lc)

    # create artists for legend
    handles = [
        plt.Line2D((0,1),(0,0), marker=markers[i], linestyle='', label=ovs, color='black', markeredgewidth='0.5')
        for i,_ in enumerate(df_optimal_filtered['Oversampler'].unique())
    ]
    labels = list(df_optimal_filtered['Oversampler'].unique())

    # ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # axes[0][0].legend(handles=handles, labels=labels,loc='lower left', numpoints=1)

    if len(classifiers) > 1:
        for hax, classifier in zip(axes, classifiers):
            hax[0].set_ylabel(classifier)
    for vax, metric in zip(axes[-1], metrics):
        metric_name = metric_names[metric] if metric in metric_names else metric
        vax.set_xlabel(metric_name)

    return fig

def plot_roc(experiment):
    figs, titles = [], []
    dataframes_per_dataset = [experiment['roc'][experiment['roc']['Dataset'] == dataset_name].reset_index(drop=True) for dataset_name in np.unique(experiment['roc']['Dataset']) ]
    for df in dataframes_per_dataset:
        col_count = 3
        row_count = math.ceil(df.shape[0] / col_count)
        fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(
            3 * col_count, 5 * row_count), sharey='row', sharex='col')
        if row_count == 1:
            axes = [axes]
        for i, row in df.iterrows():
            ax = axes[i // col_count][i % col_count]
            fpr = np.asarray(row[4:-100])
            tpr = np.asarray(row[-100:])
            ax.set_title('{} {} {}'.format(*row[:3]))
            ax.plot(fpr, tpr, lw=1)
            ax.text(.98, .98,
                'AUC: {}'.format(round(row[3], 4)),
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        unused_ax = (row_count * col_count) % df.shape[0]
        [fig.delaxes(ax) for ax in axes[-1][-unused_ax:]]
        # fig.tight_layout() # makes pdfpages empty
        title = plt.suptitle('ROC Curves: {}'.format(df.iloc[0,0]),
            fontsize=14,  y=1)
        figs.append(fig)
        titles.append(title)

    return figs, titles


def main():
    which_plot = sys.argv[2] if len(sys.argv) > 2 else None
    plot = {}
    if which_plot is not None:
        plot = {key: False for key in ['mean_cv','ranking','roc','comparison']}
        plot[which_plot] = True
    metrics = sys.argv[3].split(',') if len(sys.argv) > 3 else 'all'
    create_png(sys.argv[1], metrics=metrics, **plot)

if __name__ == "__main__":
    main()
