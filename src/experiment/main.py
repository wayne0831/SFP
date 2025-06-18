##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Experiment
# SECTION: main_real
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import json
from src.common.config import *
from src.util.experiment import *
from src.util.preprocess import OnlineStandardScaler
from src.util.metric import hit_ratio
from src.util.visualization import *
from itertools import product

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

##################################################################################################################################################################
# load and preprocess dataset
##################################################################################################################################################################

# load dataset
df_path = './data/application/df_sfp.csv'
index   = DATASET[DATA_TYPE][DATA]['INDEX']
df      = pd.read_csv(df_path, index_col = index)

print(os.environ.get('MY_ENV_VAR'))

# divide dataset into X and y
input = DATASET[DATA_TYPE][DATA][f'INPUT_{VER}']
trgt  = DATASET[DATA_TYPE][DATA]['TRGT']
X = df.loc[:, input]
y = df.loc[:, trgt]
y = y.values.ravel() # Note) y must be array!

##################################################################################################################################################################
# set experiment setting
##################################################################################################################################################################

# set initial index of training set
init_tr_start_idx   = INFRM_ADAPT[DATA_TYPE][DATA]['INIT_TR_START_IDX']
init_num_tr         = INFRM_ADAPT[DATA_TYPE][DATA]['INIT_NUM_TR']
init_tr_end_idx     = init_tr_start_idx + init_num_tr

# set problem type
prob_type = DATASET[DATA_TYPE][DATA]['PROB_TYPE']

# set performance bound for determining the prediction results as right or wrong
ctq_thr = 1 # np.std(y[:init_tr_end_idx])

run_experiment_cdda(X                = X, 
                    y                = y, 
                    scaler           = SCALER,
                    tr_start_idx     = init_tr_start_idx, 
                    tr_end_idx       = init_tr_end_idx, 
                    len_batch        = LEN_BATCH, 
                    min_len_tr       = MIN_LEN_TR, 
                    perf_bnd         = ctq_thr,
                    prob_type        = prob_type,
                    res_df_perf_path = RES_PATH['PERF_ROOT'] + RES_PATH['CDDA'], 
                    res_df_pred_path = RES_PATH['PRED_ROOT'] + RES_PATH['CDDA'])

##################################################################################################################################################################
# visualize results
##################################################################################################################################################################

res_dir_pred = RES_PATH['PRED_ROOT'] + RES_PATH['CDDA']
res_dir_perf = RES_PATH['PERF_ROOT'] + RES_PATH['CDDA']

file_name    = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH_LIST[0]}_{CDA_METH_LIST[0]}_{LEN_BATCH}_{VER}'

visualize_cdda_results(res_dir_pred=res_dir_pred, res_dir_perf=res_dir_perf, file_name=file_name)

