import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import os
import statistics as stats
from statistics import stdev
from scipy.stats import t
from math import sqrt
from config import SAMPLE_SETS

F_BETA = 1


COMPARE_TYPE_DISPLAY = {
    True: {
        'linestyle': '-',
        'marker': 'o'
    },
    False: {
        'linestyle': '--',
        'marker': '^'
    },
}

SCORE_TYPE_DISPLAY = {
    'precision': {'color': 'red'},
    'recall': {'color': 'blue'},
    'fscore': {'color': 'orange'},
}


def fbeta(precision, recall, beta=1):
    if precision == 0:
        return 0
    else:
        return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)


def get_percentile(scores_plot_values, percentile):
    res = []
    for scores_plot_value in scores_plot_values:
        res.append(np.percentile(scores_plot_value, percentile))
    return res

def get_means_stds(scores_plot_values):
    means = [
        stats.mean(scores_plot_value) for scores_plot_value in scores_plot_values
    ]
    stds = [
        stats.stdev(scores_plot_value) for scores_plot_value in scores_plot_values
    ]
    return means, stds

def plot(ax, plot_values, scores, compare_type, score_type):
    means, stds = get_means_stds(scores)
    ax.errorbar(
        plot_values,
        means,
        yerr=stds,
        label=f'{score_type} {compare_type}',
        **COMPARE_TYPE_DISPLAY[compare_type],
        **SCORE_TYPE_DISPLAY[score_type]
    )
    return means, stds


def corrected_dependent_ttest(data1, data2, n_training_folds, n_test_folds):
    n = len(data1)
    differences = [(data1[i]-data2[i]) for i in range(n)]
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    test_training_ratio = n_test_folds / n_training_folds  
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, p


def get_significance(population1, population2,  n_training_folds, n_test_folds, thresholds=(0.01, 0.05, 0.10)):
    _, _, p = corrected_dependent_ttest(population1, population2, n_training_folds, n_test_folds)
    for n, threshold in enumerate(thresholds):
        if p < threshold:
            n_asterix = len(thresholds) - n
            return n_asterix * '*'
    return ''

def main(experiment, plot_variable):

    if experiment in ('exclude', 'evaluate'):
        dfs = [
            pd.read_excel(os.path.join('results', f'{experiment}-{sample_set}.xlsx'))
            for sample_set in SAMPLE_SETS
        ]
        data = pd.concat(dfs, ignore_index=True)
        plot_type = 'hist'
        plot_values = [f'{experiment}-{sample_set}' for sample_set in SAMPLE_SETS]
    else:
        data = pd.read_excel(os.path.join('results', f'{experiment}.xlsx'))
        plot_type = 'plot'
        plot_values = sorted([v for v in set(data[plot_variable])])

    assert len(set(data['n_folds'])) == 1
    n_folds = data['n_folds'][0]
    n_training_folds = n_folds - 1
    n_validation_folds =  1

    # if plot_variable != 'l2_reg_lambda_1':
    #     data = data[data['l2_reg_lambda'] == 0.025]
    # data = data[data['n_epochs'] == 1_000]
    data = data[data['rainfall_dataset'] == 'GSMaP']
    # data = data[data['discard_below_or_equal_to_value'] == 0]

    data = data[data['test_model'] == False]
    # data = data[data['positive_labels'] == 'flood+pre-flood']
    # data = data[data['use_hydrology'] == True]

    print(f"plotting {len(data)} datapoints")

    assert len(data) != 0

    fig, ax = plt.subplots()


    all_scores = {
        'precision': {},
        'recall': {},
        'f-score': {},
        '% pos val': {}
    }
    for compare_type in (True, False):
        
        all_scores['precision'][compare_type] = {}
        all_scores['recall'][compare_type] = {}
        all_scores['f-score'][compare_type] = {}
        all_scores['% pos val'][compare_type] = {}

        precisions = []
        recalls = []
        fscores = []
        for plot_value in plot_values:
            plot_data = data[(data[plot_variable] == plot_value) & (data['use_hydrology'] == compare_type)]
            precisions.append(plot_data['precision'].to_list())
            recalls.append(plot_data['recall'].to_list())
            fscores_plot_value = [
                fbeta(p, r, beta=F_BETA)
                for p, r in zip(plot_data['precision'], plot_data['recall'])
            ]
            fscores.append(fscores_plot_value)

            all_scores['f-score'][compare_type][plot_value] = fscores_plot_value
            all_scores['precision'][compare_type][plot_value] = plot_data['precision'].to_list()
            all_scores['recall'][compare_type][plot_value] = plot_data['recall'].to_list()
            all_scores['% pos val'][compare_type][plot_value] = plot_data['% pos val'].to_list()

        plot(ax, plot_values, precisions, compare_type=compare_type, score_type='precision')
        plot(ax, plot_values, recalls, compare_type=compare_type, score_type='recall')
        plot(ax, plot_values, fscores, compare_type=compare_type, score_type='fscore')

    score_types = ['precision', 'recall', 'f-score']
    columns = ['% pos val'] + [score_type + '_pos' for score_type in score_types] + [score_type + '_neg' for score_type in score_types] + ['f1-score difference']
    df = pd.DataFrame(columns=columns, index=plot_values)

    differences = []
    for score_type in score_types:
        for plot_value in plot_values:
            positive = all_scores[score_type][True][plot_value]
            negative = all_scores[score_type][False][plot_value]
            significance = get_significance(positive, negative, n_training_folds, n_validation_folds)
            df.set_value(plot_value, score_type + '_pos',  "%.2f" % round(np.mean(positive), 2) + significance)
            df.set_value(plot_value, score_type + '_neg',  "%.2f" % round(np.mean(negative), 2) + significance)
            if score_type == 'f-score':
                difference = np.mean(positive) -  np.mean(negative)
                differences.append(difference)
                df.set_value(plot_value, 'f1-score difference',  "%.4f" % round(difference, 4))

    for plot_value in plot_values:
        percent_positive_validation = round(np.mean(all_scores['% pos val'][True][plot_value]), 2)
        df.set_value(plot_value, '% pos val', percent_positive_validation)

    print(df)
    print('average difference:', np.mean(differences))
    df.to_excel(f'results/{experiment}.xlsx')

    ax.set_ylim([0, 1.02])
    ax.set_xlim([plot_values[0], plot_values[-1]])
    ax.set_ylabel('score')
    ax.set_xlabel(plot_variable)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    for experiment, plot_variable in (
        ('compare_pos_weight', 'pos_weight'),
        ('evaluate', 'split'),
        ('exclude', 'split'),
    ):
        main(experiment, plot_variable)