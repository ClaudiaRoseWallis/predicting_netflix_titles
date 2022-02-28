'''
-------------------------------------------------------------------------------
This python file contains functions that are useful in a typical Model build
project.

Functions include:
-> data_summary
-> hex_rgb
-> down_sample
-> deciles
-> deciles_per_train
-> decile_stats
-> decile_plot
-> psi_calc
-> opt_prob_threshold
-> threshold_based_metrics

-------------------------------------------------------------------------------
'''

# import packages
import pandas as pd
import numpy as np
import time
import pickle
import os
import random
from random import sample
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.gridspec import GridSpec
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix




# functions
def data_summary(df, id_col, gb_col = '', time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # create summarise table, given gb_col parameter
    if gb_col == '':
        summ_df = df[[id_col]]
        sum_df['row_count'] = summ_df.shape[0]
        summ_df['unique_id_count'] = summ_df[id_col].nunique()
        summ_df = summ_df[['row_count',
                           'unique_id_count']].drop_dupliactes()

    else:
        summ_df = df[[id_col, gb_col]]
        summ_df['row_count'] =\
        summ_df.groupby(gb_col)[id_col].transform('count')
        summ_df['unique_id_count'] =\
        summ_df.groupby(gb_col)[id_col].transform('nunique')
        summ_df = summ_df[[gb_col, 'row_count',
                           'unique_id_count']].drop_duplicates()
        summ_df.sort_values(by = gb_col, inplace = True)

    # reset index
    summ_df.reset_index(drop = True, inplace = True)

    # give execution time of function if requested
    if time_var == True:
        print('data_summary: %s seconds' % (time.time() - start_time))

    return summ_df



def hex_rgb(hex_ls):
    rgb_ls = []
    for i in range(len(hex_ls)):
        rgb_ls.append(hex_ls[i].lstrip('#'))
        rgb_ls[i] = tuple(int(rgb_ls[i][j:j+2], 16) for j in (0, 2, 4))
        rgb_ls[i] = 'rgb' + str(rgb_ls[i])

    return rgb_ls



def down_sample(df, id_col = 'id', label_col = 'Label', date_col = '',
                majority_class_label = 0, x = 5, time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # create working df
    wdf = df.copy()

    # keep all minority class rows
    ds_df = wdf[wdf[label_col] != majority_class_label]

    # determine majority class rows to keep
    # based on overall volume
    if date_col == '':
        # determine sample number as x * minority class rows
        sample_number = math.floor(ds_df.shape[0] * x)

        # identify all majority class rows
        majority_class_ids =\
        wdf[wdf[label_col] == majority_class_label][id_col].to_list()

        # of these randomly select a predetermined volume to keep
        keep_ids = sample(majority_class_ids, sample_number)

    # based on monthly volumes
    else:
        # create col with month & year
        wdf['year_month'] = pd.to_datetime(wdf[date_col]).dt.to_period('M')

        # copy df
        monthly_class_counts = wdf.copy()

        # count rows per label & month-year
        monthly_class_counts['count'] =\
        monthly_class_counts.groupby([label_col,
                                      'year_month'])[id_col].transform('count')

        # create summary table by removing dups
        monthly_class_counts['count'] =\
        monthly_class_counts[[label_col, 'year_month',
                              'count']].drop_duplicates()

        # separate majority & minority class rows enabling merging of monthly
        # counts per class
        monthly_class_counts_majority_class =\
        monthly_class_counts[monthly_class_counts[label_col] ==\
                             majority_class_label]
        monthly_class_counts_minority_class =\
        monthly_class_counts[monthly_class_counts[label_col] !=\
                             majority_class_label]
        monthly_class_counts = pd.merge(monthly_class_counts_majority_class,
                                        monthly_class_counts_minority_class,
                                        on = 'year_month', how = 'left',
                                        suffixes = ['orig_maj_class',
                                                    'orig_min_class'])

        # clean summarised table and order by month-year column
        monthly_class_counts =\
        monthly_class_counts[['year_month', 'count_orig_maj_class',
                              'count_orig_min_class']].\
        sort_values(by = 'year_month')

        # create empty list to fill with ids of majority class to keep
        keep_ids = []

        # for each month choose majority rows to keep
        for d in monthly_class_counts['year_month'].unique():
            # determine sample number as x * minority class rows
            sample_number =\
            math.floor((monthly_class_counts.\
                        loc[monthly_class_counts['year_month'] == d,
                                                 'count_min_class'].\
                        iloc[0]) * x)

            # identify all majority class rows for the month
            majority_class_ids =\
            wdf[(wdf['year_month'] == d) &
                (wdf[label_col] == majority_class_label)][id_col].to_list()

            # of these select the predetermined volume to keep for the month
            keep_ids = keep_ids + sample(majority_class_ids, sample_number)

        # remove month year col from original df
        wdf.drop('year_month', axis = 1, inplace = True)

    # combine selected minority & majority class rows
    ds_df = ds_df.append(wdf[wdf[id_col].isin(keep_ids)])

    # add final majority class count col to monthly class
    if date_col != '':
        # get maj class rows in ds_df
        final_maj_class_counts =\
        ds_df[ds_df[label_col] == majority_class_label]

        # add month-year col
        final_maj_class_counts['year_month'] =\
        pd.to_datetime(final_maj_class_counts[date_col]).dt.to_period('M')

        # count rows per month year
        final_maj_class_counts['count_final_maj_class'] =\
        final_maj_class_counts.groupby('year_month')[id_col].transform('count')

        # create summary table by removing dups
        final_maj_class_counts =\
        final_maj_class_counts[['year_month',
                                'count_final_maj_class']].drop_duplicates()

        # merge final maj class count col
        monthly_class_counts = pd.merge(monthly_class_counts,
                                        final_maj_class_counts,
                                        on = 'year_month', how = 'left')

        # reset index
        monthly_class_counts.reset_index(drop = True, inplace = True)


    # reset index
    ds_df.reset_index(drop = True, inplace = True)

    # give execution time of function if requested
    if time_var:
        print('down_sample: %s seconds' % (time.time() - start_time))


    if date_col == '':
        return ds_df
    else:
        return ds_df, monthly_class_counts



def deciles(df, score_col = 'label_1_prob', weight_col = '', time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # calculate decile column to include in df
    # where df doesn't have weights
    if weight_col == '':
        # sort model scores
        df = df.sort_values(by = [score_col], ascending = False)

        # calculate deciles on the length
        F_t = np.linspace(1/len(df), 1, len(df))

        # create decile col
        df['decile'] = np.ceil(10 * F_t)

    # where df has weights
    else:
        # define values
        values = df[score_col]

        # define weights
        weights = df[weight_col]

        # define number of quantiles
        quantiles = 10

        # get weighted quantile cuts
        quantiles = np.linspace(0, 1, quantiles + 1)
        values = -1 * values
        order = weights.iloc[values.argsort()].cumsum()
        bins = pd.cut(order/order.iloc[-1], quantiles, labels = False)

        # create weighted decile col
        df['decile'] = bins.sort_index() + 1

    # reset index
    df.reset_index(drop = True, inplace = True)

    # give execution time of function if requested
    if time_var:
        print('deciles: %s seconds' % (time.time() - start_time))


    return df



def deciles_per_train(df, df_deciles, score_col = 'label_1_prob', time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # copy df_deciles
    decile_bands = df_deciles.copy()

    # create minimum score col, per decile
    decile_bands['min_score'] =\
    decile_bands.groupby('decile')[score_col].transform('min')

    # create min score per decile summary table by dropping dups
    decile_bands =\
    decile_bands[['decile',
                  'min_score']].drop_duplicates().sort_values(by = 'decile')

    # save min score per decile summary table to return
    decile_bands_min_score = decile_bands.copy()

    display(decile_bands_min_score)

    # create decile column in df and set to 1
    df['decile'] = 1

    # deciles 2 - 10
    for i in range(1, 10):
        df.loc[(df[score_col] < decile_bands[decile_bands['decile'] ==\
                                             i]['min_score'].item()) &
               (df[score_col] >= decile_bands[decile_bands['decile'] ==\
                                              i + 1]['min_score'].item()),
               'decile'] = i + 1

        if i == 9:
            df.loc[(df[score_col] < decile_bands[decile_bands['decile'] ==\
                                                 i]['min_score'].item()),
                       'decile'] = i + 1

    # reset index
    df.reset_index(drop = True, inplace = True)
    decile_bands_min_score.reset_index(drop = True, inplace = True)

    # give execution time of function if requested
    if time_var:
        print('deciles_per_train: %s seconds' % (time.time() - start_time))


    return df, decile_bands_min_score



def decile_stats(df, outcome_label = 1, score_col = 'label_1_prob',
                 label_col = 'Label', decile_col = 'decile',
                 weight_col = '', decile_cut_off = 1, print_f = True,
                 score_delta_calc = True, time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # identify non outcome label
    labels = df[label_col].unique().tolist()
    non_outcome_label =\
    [label for label in labels if label != outcome_label][0]

    # classification label volumes & portions
    if print_f:
        print(str(df[df[label_col] == outcome_label].shape[0]) + '(' +
              str(round(df[df[label_col] ==\
                           outcome_label].shape[0] / df.shape[0] * 100, 2)) +
              '%) entries are labelled ' + str(outcome_label) + '\n' +
              str(df[df[label_col] != outcome_label].shape[0]) + '(' +
              str(round(df[df[label_col] !=\
                           outcome_label].shape[0] / df.shape[0] * 100, 2)) +
              '%) entries are labelled ' + str(non_outcome_label) + '\n' +
              'There are ' + str(df.shape[0]) + ' entries in total')

    # copy df
    decile_bands = df.copy()

    # calculate score stats per decile
    decile_bands['min_score'] =\
    decile_bands.groupby(decile_col)[score_col].transform('min')
    decile_bands['max_score'] =\
    decile_bands.groupby(decile_col)[score_col].transform('max')
    decile_bands['avg_score'] =\
    decile_bands.groupby(decile_col)[score_col].transform('mean')
    decile_bands['volume'] =\
    decile_bands.groupby(decile_col)[score_col].transform('count')
    decile_bands['outcome_volume'] =\
    decile_bands.groupby([decile_col, label_col])[score_col].transform('count')

    if weight_col != '':
        decile_bands['volume_w'] =\
        decile_bands.groupby(decile_col)[weight_col].transform('sum')
        decile_bands['outcome_volume_w'] =\
        decile_bands.groupby([decile_col,
                              label_col])[weight_col].transform('sum')

        decile_bands['score_w'] =\
        decile_bands[score_col] * decile_bands[weight_col]
        decile_bands['score_sum_w'] =\
        decile_bands.groupby(decile_col)['score_w'].transform('sum')
        decile_bands['avg_score_w'] =\
        decile_bands[score_sum_w] / decile_bands[volume_w]

    # remove dups with a simple decile band table
    if weight_col != '' :
        decile_band_base =\
        decile_bands[[decile_col, 'min_score', 'max_score',
                      'avg_score', 'avg_score_w', 'volume',
                      'volume_w']].drop_duplicates()
        outcome_volumes =\
        decile_bands[decile_bands[label_col] ==\
                     outcome_label][[decile_col, 'outcome_volume',
                                     'outcome_volume_w']].\
        drop_duplicates().sort_values(by = decile_col)

    else:
        decile_band_base =\
        decile_bands[[decile_col, 'min_score', 'max_score', 'avg_score',
                      'volume']].drop_duplicates()
        outcome_volumes =\
        decile_bands[decile_bands[label_col] ==\
                                  outcome_label][[decile_col,
                                                  'outcome_volume']].\
        drop_duplicates().sort_values(by = decile_col)

    # merge outcome_volumes to decile_band_base
    ## as long as each decile has some value of either label each decile will
    ## be represented
    decile_bands = pd.merge(decile_band_base, outcome_volumes, how = 'left',
                            on = decile_col).fillna(0)


    # calculate outcome rates
    decile_bands['outcome_rate'] =\
    decile_bands['outcome_volume'] / decile_bands['volume']
    if weight_col != '':
        decile_bands['outcome_rate_w'] =\
        decile_bands['outcome_volume_w'] / decile_bands['volume_w']


    # calculate log odds and, subsequently, score deltas if relevant
    if score_delta_calc:
        if weight_col != '':
            decile_bands['actual_log_odds'] =\
            np.log(decile_bands['outcome_rate_w'] /\
                   (1 - decile_bands['outcome_rate_w']))
            decile_bands['predicted_log_odds'] =\
            np.log(decile_bands['avg_score_w'] /\
                   (1 - decile_bands['avg_score_w']))
        else:
            decile_bands['actual_log_odds'] =\
            np.log(decile_bands['outcome_rate'] /\
                   (1 - decile_bands['outcome_rate']))
            decile_bands['predicted_log_odds'] =\
            np.log(decile_bands['avg_score'] /\
                   (1 - decile_bands['avg_score']))

    decile_bands['score_delta'] =\
    abs(decile_bands['actual_log_odds'] - decile_bands['predicted_log_odds'])


    # order summary table
    decile_bands = decile_bands.sort_values(by = decile_col)


    # print % of outcome ids in top x deciles specified by decile_cut_off
    # parameter
    if print_f:
        print(str(round(decile_bands[decile_bands[decile_col] <=\
                                     decile_cut_off]['outcome_volume'].sum() /\
                        df[df[label_col] ==\
                           outcome_label].shape[0] * 100, 2)) +
              '% of outcomes in top ' + tr(decile_cut_off) + ' deciles\n')


    # calculate overall score delta if relevant
    if score_delta_calc:
        # exclude deciles with 0 outcomes from this calc to avoid inf score
        # delta
        sd_decile_bands = decile_bands[decile_bands['outcome_volume'] != 0]

        # calc overall Score Delta, ideally this is < 0.5 if scores and outcome
        # rates are aligned
        if weight_col != '':
            sd_decile_bands['product'] =\
            sd_decile_bands['volume_w'] * sd_decile_bands['score_delta']
            sd =\
            (sd_decile_bands['product'].sum() /\
             sd_decile_bands['volume_w'].sum()) / math.log(2)
            if print_f:
                print('Score Delta = ' + str(round(sd, 2)))
        else:
            sd_decile_bands['product'] =\
            sd_decile_bands['volume'] * sd_decile_bands['score_delta']
            sd =\
            (sd_decile_bands['product'].sum() /\
             sd_decile_bands['volume'].sum()) / math.log(2)
            if print_f:
                print('Score Delta = ' + str(round(sd, 2)))

    # reset index
    decile_bands.reset_index(drop = True, inplace = True)

    # give execution time of function if requested
    if time_var:
        print('decile_stats: %s seconds' % (time.time() - start_time))


    return decile_bands



def decile_plot(decile_bands, outcome_rate_col = 'outcome_rate',
                decile_col = 'decile',
                title = 'Outcome Rate per Decile (Train)',
                y_title = 'Outcome Rate (%)', x_title = 'Decile',
                figsize_x = 15, figsize_y = 7, title_font_size = 25,
                axis_title_font_size = 20, tick_size = 15, title_gap = 1.05,
                bar_text_font_size = 15, colour = '#ffcc00', time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # set plot params
    sns.set(style = 'darkgrid')
    plt.figure(figsize = (figsize_x, figsize_y))

    # convert outcome rate to percentage
    decile_bands['percent'] = round(decile_bands[outcome_rate_col] * 100, 2)

    # remove any excess cols from decile_bands
    decile_bands = decile_bands[[decile_col, 'percent']]

    # account for potential missing deciles
    base = pd.DataFrame({decile_col : list(range(1, 11))})
    decile_bands = pd.merge(base, decile_bands, how = 'left',
                            on = decile_col).fillna(0)

    # create plot
    ax = sns.barplot(x = decile_col, y = 'percent', data = decile_bands,
                     estimator = sum, ci = None, color = colour)
    ax.set_title(title, fontsize = title_font_size, y = title_gap)
    ax.set_ylabel(y_title, fontsize = axis_title_font_size)
    ax.set_xlabel(x_title, fontsize = axis_title_font_size)
    plt.yticks(size = tick_size)
    plt.xticks(size = tick_size)
    plt.tight_layout()

    # label each bar with percentage
    for p in ax.patches:
        height = p.get_height
        ax.text(x = p.get_x() + (p.get_width()/2),
                y = p.get_height() + 0.005, s = p.get_height(),
                ha = 'center', size = bar_text_font_size)

    # give execution time of function if requested
    if time_var:
        print('decile_plot: %s seconds' % (time.time() - start_time))

    return plt



def psi_calc(db_orig, db_compare, decile_col = 'decile', volume_col = 'volume',
             orig_suffix = 'train', compare_suffix = 'test', time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # merge db_orig & db_compare to compare volumes per decile
    df =\
    pd.merge(db_orig[[decile_col, volume_col]],
             db_compare[[decile_col, volume_col]],
             how = 'left', on = decile_col,
             suffixes = ['_' + orig_suffix, '_' + compare_suffix]).fillna(0)

    # calc ratio of ids per decile
    df[orig_suffix + '_ratio'] =\
    round(df[volume_col + '_' + orig_suffix] /\
    df[volume_col + '_' + orig_suffix].sum(), 5)
    df[compare_suffix + '_ratio'] =\
    round(df[volume_col + '_' + compare_suffix] /\
    df[volume_col + '_' + compare_suffix].sum(), 5)

    # convert ratios to percentages
    df[orig_suffix + '_percent'] = df[orig_suffix + '_ratio'] * 100
    df[compare_suffix + '_percent'] = df[compare_suffix + '_ratio'] * 100

    # calculate PSI per decile
    df['psi'] = round((df[orig_suffix + '_ratio'] -
                       df[compare_suffix + '_ratio']) * \
                       np.log(df[orig_suffix + '_ratio'] /\
                              df[compare_suffix + '_ratio']), 5)

    # calculate overall PSI such that inf psi deciles aren't included in calc
    # and print note psi < 0.1 indicates population is stable
    df['no_inf_psi'] = df['psi']
    df['no_inf_psi'].replace(np.inf, 0, inplace = True)
    psi = df['no_inf_psi'].sum()
    print('Overall ' + compare_suffix + ' PSI = ' + str(psi))

    # specify columns to keep
    df =\
    df[[decile_col, volume_col + '_' + orig_suffix, orig_suffix + '_percent',
        volume_col + '_' + compare_suffix, compare_suffix + '_percent', 'psi']]

    # reset index
    df.reset_index(drop = True, inplace = True)

    # give execution time of function if requested
    if time_var:
        print('psi_calc: %s seconds' % (time.time() - start_time))


    return df, psi



def opt_prob_threshold(test_probs, Y, colour, metric = 'accuracy',
                       intervals = 0.1, plot = True, time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # calculate metric score for each potential threshold choice and store in
    # data frame
    thresholds = []
    metrics = []
    for i in list(np.arange(0, 1, intervals)):
        thresholds.append(i)
        pred_i = (test_probs[:, 1] >= i).astype(bool)

        if metric == 'accuarcy':
            metrics.append(accuracy_score(Y, pred_i))
        if metric == 'f1':
            metrics.append(f1_score(Y, pred_i))

    threshold_df = pd.DataFrame(dict(threshold = thresholds, metric = metrics))

    # get max metric values
    max_met = threshold_df['metric'].max()
    opt_threshold =\
    threshold_df[threshold_df['metric'] == max_met]['threshold'].tolist()[0]

    # plot
    if plot == True:
        cf.set_config_file(dimensions = (800, 800), margin = (70, 70, 70, 70))
        threshold_df.iplot(x = 'threshold', y = 'metric', colors = colour,
                           title = 'Probability Thresholds by ' + metric,
                           xTitle = 'Threshold', yTitle = metric)

    # give execution time of function if requested
    if time_var:
        print('opt_prob_threshold: %s seconds' % (time.time() - start_time))

    return threshold_df, opt_threshold



def threshold_based_metrics(df, actual_label_col = 'label', threshold = 0.5,
                            score_col = 'label_1_prob',
                            predicted_label_col = 'prediction',
                            outcome_label = 1, title = 'Confusion Matrix',
                            print_f = True, time_var = False):
    # save function start time to calc function time later
    start_time = time.time()

    # identify non outcome label
    labels = df[actual_label_col].unique().tolist()
    non_outcome_label =\
    [label for label in labels if label != outcome_label][0]

    # create prediction_label_col based on given threshold
    df[predicted_label_col] = non_outcome_label
    df.loc[df[score_col] >= threshold, predicted_label_col] = outcome_label

    # print classification report
    if print_f:
        print(classification_report(df[actual_label_col],
                                    df[predicted_label_col]))

    # cross tabulate data
    pd.crosstab(df[actual_label_col], df[predicted_label_col],
                rownames = ['Actual'], colnames = ['Predicted'],
                margins = True)

    # define confusion matrix
    cm = confusion_matrix(df[actual_label_col], df[predicted_label_col],
                          labels = labels)
    df_cm = pd.DataFrame(cm)

    # set plot details using gridspec
    fig = plt.figure(figsize = (19, 4), constrained_layout = True)
    gs = GridSpec(ncols = 19, nrows = 1, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0:5])
    ax2 = fig.add_subplot(gs[0, 7:12])
    ax3 = fig.add_subplot(gs[0, 14:19])
    fig.suptitle(title, size = 25, y = 1.2)

    # set labels
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    group_names = ['True Neg', 'False Pos', 'False Neg', 'Trie Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]

    # percents based on total volumes
    total = tn + fp + fn + tp
    percents = [tn/total, fp/total, fn/total, tp/total]
    group_percentages = ['{0:.2%}'.format(value) for value in percents]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,
                                                        group_counts,
                                                        group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax1.set(xlabel = 'Predicted', ylabel = 'Actual')
    ax1.set_title(label = 'Overall Percents', size = 14)
    sns.heatmap(df_cm, annot = labels, fmt = '', ax = ax1)
    # use cmap to specify palette

    # percents based on actual volumes
    percents = [tn/(tn + fp), fp/(tn + fp), fn/(fn + tp), tp/(fn + tp)]
    group_percentages = ['{0:.2%}'.format(value) for value in percents]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,
                                                        group_counts,
                                                        group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax2.set(xlabel = 'Predicted', ylabel = 'Actual')
    ax2.set_title(label = 'Percents per Actual Label', size = 14)
    sns.heatmap(df_cm, annot = labels, fmt = '', ax = ax2)
    # use cmap to specify palette

    # percents based on actual volumes
    percents = [tn/(tn + fn), fp/(tp + fp), fn/(fn + tn), tp/(fp + tp)]
    group_percentages = ['{0:.2%}'.format(value) for value in percents]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,
                                                        group_counts,
                                                        group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax3.set(xlabel = 'Predicted', ylabel = 'Actual')
    ax3.set_title(label = 'Percents per Predicted Label', size = 14)
    sns.heatmap(df_cm, annot = labels, fmt = '', ax = ax3)
    # use cmap to specify palette

    # print true & false positive rates
    if print_f:
        print('True positive rate (sensitivity/recall) = ' +
              str(round(tp / (tp + tn) * 100, 2)) + '%' +
              '\nFalse positive rate = ' +
              str(round(fp / (fp + tn) * 100, 2)) +'%' +
              '\nPrecision = ' + str(round(tp / (fp + tp) * 100, 2)) + '%' +
              '\nFalse discovery rate = ' +
              str(round(fp / (fp + tp) * 100, 2)) + '%' +
              '\nFalse negative rate = ' +
              str(round(fn / (tp + fn) * 100, 2)) + '%' +
              '\nTrue negative rate = ' +
              str(round(tn / (fp + tn) * 100, 2)) + '%')

    # give execution time of function if requested
    if time_var:
        print('threshold_based_metrics: %s seconds' % (time.time() -
                                                       start_time))

    return fig
