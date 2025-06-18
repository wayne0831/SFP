##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Util
# SECTION: Experiment
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import json

from src.common.config import *
from src.util.metric import hit_ratio
from itertools import product

from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import pandas as pd
import numpy as np

##################################################################################################################################################################
# set user-defined functions
##################################################################################################################################################################

def generate_parameter_combinations(CDD_PARAM_GRID, CDD_METH):
    """
    Generate all hyperparameter combinations

    :param  CDD_PARAM_GRID: param grid of cdd method
    :param  CDD_METH: cdd method
    :return param_comb_list: list of hyperparameter combinations
    """
    # generate combinations of hyperparameter
    cdd_params  = list(CDD_PARAM_GRID[CDD_METH].keys())    # ex) ['alpha_w', 'alpha_d'...]
    cdd_values  = list(CDD_PARAM_GRID[CDD_METH].values())  # ex) [[1.5, 2], [2.5, 3], ...]
    param_comb  = product(*cdd_values)                     # ex) [[1.5 , 2.5], [1.5, 3], ...]

    param_comb_list = []
    for comb in param_comb:
        param = dict(zip(cdd_params, comb)) # ex) {'alpha_w': 1.5, 'alpha_d': 2.5, ..}
        param_comb_list.append(param)       # ex) [{'alpha_w': 1.5, 'alpha_d': 2.5, ..}, {...}, ...]
    # end for

    return param_comb_list

def run_experiment_cdda(X, y, scaler, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd, prob_type,
                        res_df_perf_path, res_df_pred_path):
    
    for CDD_METH, CDA_METH in product(CDD_METH_LIST, CDA_METH_LIST):
        print('*' * 150)
        print(f'ML_METH: {ML_METH}, CDD_METH: {CDD_METH}, CDA_METH: {CDA_METH}')

        res_df_perf = pd.DataFrame()
        res_df_pred = pd.DataFrame()
    
        param_comb_list = generate_parameter_combinations(CDD_PARAM_GRID=CDD_PARAM_GRID, CDD_METH=CDD_METH)

        for param_comb in param_comb_list:
            #print(f'param_comb: {param_comb}')
            start_time = time.time()

            cdda = CDD[CDD_METH](**param_comb)
            cdda.run_cdda(X             = X, 
                          y             = y, 
                          scaler        = scaler, 
                          prob_type     = prob_type, 
                          ml_mdl        = ML[PROB_TYPE][ML_METH],
                          tr_start_idx  = tr_start_idx,
                          tr_end_idx    = tr_end_idx,
                          len_batch     = len_batch,
                          min_len_tr    = min_len_tr,
                          perf_bnd      = perf_bnd)
            
            end_time = time.time()

            time_idx    = cdda.res_cdda['time_idx']
            y_real_list = cdda.res_cdda['y_real_list']
            y_pred_list = cdda.res_cdda['y_pred_list']

            if PROB_TYPE == 'REG':
                res_perf_idx = {
                    'cdd_method':      str(CDD_METH),
                    'mape':      np.round(mean_absolute_percentage_error(y_real_list, y_pred_list) * 100, 4),
                    'mae':       np.round(mean_absolute_error(y_real_list, y_pred_list), 4),
                    'rmse':      np.round(root_mean_squared_error(y_real_list, y_pred_list), 4),
                    'r2':        np.round(r2_score(y_real_list, y_pred_list), 4),
                    'ctq':       np.round(hit_ratio(y_real_list, y_pred_list, perf_bnd) * 100, 2),
                    'num_cd':    len(cdda.res_cdda['cd_idx']),
                    'exec_time(s)': np.round((end_time - start_time), 2),
                    'init_tr_end_idx': tr_end_idx,
                    'cd_idx':    cdda.res_cdda['cd_idx'], 
                }
            elif PROB_TYPE == 'CLF':
                res_perf_idx = {
                    'cdd_method':      str(CDD_METH),
                    'param':           json.dumps(param_comb),
                    'init_tr_end_idx': tr_end_idx,
                    'cd_idx':    cdda.res_cdda['cd_idx'], 
                    'num_cd':    len(cdda.res_cdda['cd_idx']),
                    'len_adapt': cdda.res_cdda['len_adapt'],
                    'acc':       np.round(accuracy_score(y_real_list, y_pred_list) * 100, 2),
                }                
            
            print('Experiment Results')
            filtered = {k: v for k, v in res_perf_idx.items() if k not in {'init_tr_end_idx', 'cd_idx'}}
            print(filtered)

            res_pred_idx = {
                'cdd_method':   str(CDD_METH),
                'param':    json.dumps(param_comb),                
                'time_idx': time_idx,
                'y_real':   y_real_list,
                'y_pred':   y_pred_list,
            }

            res_df_perf  = pd.concat([res_df_perf, pd.DataFrame([res_perf_idx])], ignore_index = True)
            res_df_pred  = pd.concat([res_df_pred, pd.DataFrame(res_pred_idx)], ignore_index = True)
        # end for param_comb

        # set prediction result dataframe
        res_df_pred_name = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH}_{CDA_METH}_{LEN_BATCH}_{VER}_PRED.csv'
        res_df_pred.to_csv(res_df_pred_path + res_df_pred_name)

        # set performance results
        res_df_perf_name = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH}_{CDA_METH}_{LEN_BATCH}_{VER}_PERF.csv'
        res_df_perf.to_csv(res_df_perf_path + res_df_perf_name)
    
    # end for param_comb ML_METH, CDD_METH, CDA_METH

    return None