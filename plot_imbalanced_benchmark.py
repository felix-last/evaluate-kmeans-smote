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
from collections import Counter
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import imbtools.evaluation
from imbtools.evaluation import calculate_stats, calculate_optimal_stats, calculate_optimal_stats_wide, calculate_ranking, calculate_mean_ranking, calculate_friedman_test
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

def create_pdf(session_id, friedman_test=True, mean_cv=True, roc=True):
    path = cfg['results_dir']

    experiment = load_experiment(session_id)

    with PdfPages('{0}/{1}/{1}.pdf'.format(path, session_id)) as pdf:
        if friedman_test and 'friedman_test_results' in experiment:
            fig, title = plot_mean_ranking(experiment['mean_ranking_results'], experiment['friedman_test_results'])
            pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")
        if mean_cv and 'mean_cv_results' in experiment:
            fig, title = plot_cross_validation_mean_results(experiment['mean_cv_results'], experiment['std_cv_results'])
            pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")
        if roc and 'roc' in experiment:
            figs, titles = plot_roc(experiment)
            for fig, title in zip(figs, titles):
                pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")

def load_experiment(session_id):
    path = cfg['results_dir']
    if 'experiment.p' in os.listdir( os.path.join(path, session_id) ):
        # new imbtools 1.0.0 format
        binary_experiment = imbtools.evaluation.load_experiment('{}/{}/experiment.p'.format(path, session_id))
        optimal_stats = calculate_optimal_stats(binary_experiment)
        experiment = {
            'mean_ranking_results': calculate_mean_ranking(binary_experiment),
            'friedman_test_results': calculate_friedman_test(binary_experiment),
            'mean_cv_results': optimal_stats.drop('Std CV score', axis=1),
            'std_cv_results': optimal_stats.drop('Mean CV score', axis=1),
        }
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
    return experiment

def plot_mean_ranking(mean_ranking_results, friedman_test_results):
    metrics = np.unique(mean_ranking_results['Metric'])
    classifiers = np.unique(mean_ranking_results['Classifier'])
    row_count = len(classifiers)
    col_count = len(metrics)
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(
        3 * col_count, 5 * row_count), sharey='row')
    if row_count == 1:
        axes = [axes]

    for j, metric in enumerate(metrics):
        axes[0][j].set_title(metric)

    for i, classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            ranking_filtered = mean_ranking_results
            ranking_filtered = ranking_filtered[
                (ranking_filtered['Classifier'] == classifier)
                & (ranking_filtered['Metric'] == metric)
            ]
            ranking_filtered = ranking_filtered.drop(
                ['Classifier', 'Metric'], axis=1)
            ax = axes[i][j]
            methods_encoded = np.asarray(
                [i for i, _ in enumerate(ranking_filtered.columns)])
    #         method_colors = sns.husl_palette(n_colors=len(methods_encoded))
            ax.bar(
                methods_encoded,
                np.asarray(ranking_filtered).transpose(),
                0.5,
                #             color=method_colors
            )
            ax.set_xticks(methods_encoded + 0.25)
            ax.set_xticklabels(ranking_filtered.columns, rotation='vertical')
            if j is 0:
                ax.set_ylabel(classifier)

            corresponding_friedman = round(np.asarray(friedman_test_results[
                (friedman_test_results
                 ['Classifier'] == classifier)
                & (friedman_test_results['Metric'] == metric)
            ]['p-value'])[0], 2)
            ax.text(.98, .98, 'p-value: {}'.format(corresponding_friedman),
                    transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    fig.tight_layout()
    title = plt.suptitle(
        'Mean Ranking Results + Friedman Test', fontsize=14, y=1.02)
    return fig, title


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
        axes[0][j].set_title(metric)
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
    create_pdf(sys.argv[1])

if __name__ == "__main__":
    main()
