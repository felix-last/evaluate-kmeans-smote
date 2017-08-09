# coding: utf-8
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import yaml
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

def createPdf(session_id):
    path = cfg['results_dir']

    experiment = {}
    experiment['datasets_summary_'] = pd.read_csv('{}/{}/datasets_summary.csv'.format(path, session_id))
    experiment['friedman_test_results_'] = pd.read_csv('{}/{}/friedman_test_results.csv'.format(path, session_id))
    experiment['mean_cv_results_'] = pd.read_csv('{}/{}/mean_cv_results.csv'.format(path, session_id))
    experiment['mean_ranking_results_'] = pd.read_csv('{}/{}/mean_ranking_results.csv'.format(path, session_id))
    experiment['std_cv_results_'] = pd.read_csv('{}/{}/std_cv_results.csv'.format(path, session_id))

    with PdfPages('{}/{}/results.pdf'.format(path, session_id)) as pdf:
        fig, title = plot_mean_ranking(experiment)
        pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")
        fig, title = plot_mean_ranking(experiment)
        pdf.savefig(fig, bbox_extra_artists=(title,), bbox_inches="tight")

def plot_mean_ranking(experiment):
    metrics = np.unique(experiment['mean_ranking_results_']['Metric'])
    classifiers = np.unique(experiment['mean_ranking_results_']['Classifier'])
    row_count = len(classifiers)
    col_count = len(metrics)
    fig,axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(10,5*row_count), sharey='row')
    if row_count == 1: axes = [axes]

    for j,metric in enumerate(metrics):
        axes[0][j].set_title(metric)

    for i, classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            ranking_filtered = experiment['mean_ranking_results_']
            ranking_filtered = ranking_filtered[
                (ranking_filtered['Classifier']==classifier)
                & (ranking_filtered['Metric']==metric)
            ]
            ranking_filtered = ranking_filtered.drop(['Classifier', 'Metric'], axis=1)
            ax = axes[i][j]
            methods_encoded = np.asarray([i for i,_ in enumerate(ranking_filtered.columns)])
    #         method_colors = sns.husl_palette(n_colors=len(methods_encoded))
            ax.bar(
                methods_encoded,
                np.asarray(ranking_filtered).transpose(),
                0.5,
    #             color=method_colors
            )
            ax.set_xticks(methods_encoded + 0.25)
            ax.set_xticklabels(ranking_filtered.columns, rotation='vertical')
            if j is 0: ax.set_ylabel(classifier)

            corresponding_friedman = round(np.asarray(experiment['friedman_test_results_'][
                (experiment['friedman_test_results_']['Classifier'] == classifier)
                & (experiment['friedman_test_results_']['Metric'] == metric)
            ]['p-value'])[0], 2)
            ax.text(.98,.98,'p-value: {}'.format(corresponding_friedman), transform = ax.transAxes, horizontalalignment='right', verticalalignment='top')
    fig.tight_layout()
    title = plt.suptitle('Mean Ranking Results + Friedman Test', fontsize=14, y=1.02)
    return fig, title

def plot_cross_validation_mean_results(experiment):
    datasets = np.unique(experiment['mean_cv_results_']['Dataset'])
    classifiers = np.unique(experiment['mean_cv_results_']['Classifier'])
    metrics = np.unique(experiment['mean_cv_results_']['Metric'])

    row_count = len(classifiers)*len(datasets)
    col_count = len(metrics)
    fig,axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(10,5*row_count), sharey='row')
    if row_count == 1: axes = [axes]
    for j,metric in enumerate(metrics):
        axes[0][j].set_title(metric)
    for i, classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            for k, dataset in enumerate(datasets):
                mean_filtered = experiment['mean_cv_results_']
                mean_filtered = mean_filtered[
                    (mean_filtered['Classifier']==classifier)
                    & (mean_filtered['Metric']==metric)
                    & (mean_filtered['Dataset']==dataset )
                ]
                mean_filtered = mean_filtered.drop(['Classifier', 'Metric', 'Dataset', 'Unnamed: 0'], axis=1)

                std_filtered = experiment['std_cv_results_']
                std_filtered = std_filtered[
                    (std_filtered['Classifier']==classifier)
                    & (std_filtered['Metric']==metric)
                    & (std_filtered['Dataset']==dataset )
                ]
                std_filtered = std_filtered.drop(['Classifier', 'Metric', 'Dataset', 'Unnamed: 0'], axis=1)

                ax = axes[len(datasets)*i+k][j]
                methods_encoded = np.asarray([i for i,_ in enumerate(mean_filtered['Oversampling method'])])
    #             method_colors = sns.husl_palette(n_colors=len(methods_encoded))
                bars = ax.bar(
                    methods_encoded,
                    mean_filtered['Mean CV score'],
                    0.5,
    #                 color= method_colors,
                    yerr=std_filtered['Std CV score'],
                    ecolor='black'
                )
                ax.set_ylim((0,1))

                ax.set_xticks(methods_encoded+0.25)
                ax.set_xticklabels(mean_filtered['Oversampling method'], rotation='vertical')

                if j is 0: ax.set_ylabel('{} - {}'.format(classifier, dataset))
    # axes[0][0].legend(bars, np.asarray(mean_filtered['Oversampling method']))
    fig.tight_layout()
    title = plt.suptitle('Mean Cross-Validation Result + Standard Deviation', fontsize=14,  y=1.02)
    return fig, title
