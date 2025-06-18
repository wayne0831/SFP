##################################################################################################################################################################
# PROJECT: TM_MLOps
# CHAPTER: Util
# SECTION: Experiment
# AUTHOR: Yang et al.
# DATE: since 24.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from src.common.config import *
from src.util import *
#from algorithm.ml.ML import *

import pandas as pd
import numpy as np
import time
import os

import ast
import matplotlib.pyplot as plt

##################################################################################################################################################################
# set user-defined functions
##################################################################################################################################################################

### TODO
def plot_performances(train_len : int, in_sample_error : float, *args, colors = None, legends = None, title : str):
    if colors is None:
        colors = plt.cm.get_cmap('tab10', len(args))
    
    if legends is None:
        legends = [f'List {i+1}' for i in range(len(args))]
    
    in_sample_error_ls = [in_sample_error] * train_len
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Data index')
    ax.plot(list(range(1, train_len + 1)), in_sample_error_ls, 'g--')
    for i, y_values in enumerate(args):
        color = colors(i) if callable(colors) else colors[i % len(colors)]
        ax.plot(list(range(train_len + 1, train_len + len(y_values) + 1)),
                y_values, color=color, label = legends)
    
    plt.legend(legends)
    plt.title(title)
    plt.show()


def visualize_cdda_results(res_dir_pred, res_dir_perf, file_name):

    res_dir_pred = RES_PATH['PRED_ROOT'] + RES_PATH['CDDA']
    res_dir_perf = RES_PATH['PERF_ROOT'] + RES_PATH['CDDA']

    res_df_pred = pd.read_csv(f'{res_dir_pred}{file_name}_PRED.csv')
    res_df_perf = pd.read_csv(f'{res_dir_perf}{file_name}_PERF.csv')

    init_tr_end_idx  = res_df_perf['init_tr_end_idx'][0]
    mdl_upd_idx_list = ast.literal_eval(res_df_perf['cd_idx'][0])
    res_df_pred['time'] = pd.to_datetime(res_df_pred['time_idx'])

    plt.figure(figsize=(15, 6))
    plt.plot(res_df_pred['time'], res_df_pred['y_real'], label='Real', color='black', alpha=1)
    plt.plot(res_df_pred['time'], res_df_pred['y_pred'], label='Pred', color='red', alpha=0.9)
    plt.legend(loc='lower right')
    plt.ylabel('Target')
    for idx in mdl_upd_idx_list:
        plt.axvline(x=res_df_pred['time'][idx-init_tr_end_idx], linestyle='--', alpha=0.3)
    # for
    plt.show()

    return None