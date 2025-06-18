##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Util
# SECTION: Common
# AUTHOR: Yang et al.
# DATE: since 25.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from sklearn.preprocessing import StandardScaler
from src.util.preprocess import OnlineStandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import time

##################################################################################################################################################################
# user-defined utility functions
##################################################################################################################################################################

# def run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl):
#     # scale dataset
#     if scaler is not None:
#         X_tr = scaler.fit_transform(X_tr)
#         X_te = scaler.transform(X_te)
#     # end if

#     # train the model
#     ml_mdl.fit(X_tr, y_tr)

#     # predict testset 
#     y_pred_te = ml_mdl.predict(X_te)

#     return ml_mdl, y_pred_te

def set_ml_dataset(tr_start_idx, tr_end_idx, te_start_idx, te_end_idx, X, y):
    """
    Set dataset for ml model
    """
    tr_idx_list = list(range(tr_start_idx, tr_end_idx))
    te_idx_list = list(range(te_start_idx, te_end_idx)) # TODO: te_idx_list, te_end_idx 비교 잘하자

    # set training/test set
    X_tr, y_tr = X.iloc[tr_idx_list], y[tr_idx_list]
    X_te, y_te = X.iloc[te_idx_list], y[te_idx_list]

    return X_tr, y_tr, X_te, y_te

def run_ml_model(X_cum, y_cum, X_tr, y_tr, X_te, y_te, y, scaler, ml_mdl, prob_type, perf_bnd):
    """
    Run ml model pipeline: scale dataset, train ml model, predict data
    """

    # if online learning is available
    if isinstance(scaler, OnlineStandardScaler) and hasattr(ml_mdl, 'partial_fit'):
        # partially scale dataset
        scaler.partial_fit(X_tr)
        X_tr_scl = scaler.fit_transform(X_tr)
        X_te_scl = scaler.transform(X_te)

        # partially fit the ml model
        # y = np.concatenate([y_tr, y_te])
        ml_mdl.partial_fit(X_tr_scl, y_tr) if prob_type == 'REG' else \
        ml_mdl.partial_fit(X_tr_scl, y_tr, classes=np.unique(y))

        # predict the test sets
        y_pred_te = ml_mdl.predict(X_te_scl)
    else:
        print(f'Length of X_cum: {len(X_cum)}')
        # offline scaling
        X_tr_scl = scaler.fit_transform(X_cum)
        X_te_scl = scaler.transform(X_te)
        
        # train the model
        ml_mdl.fit(X_tr_scl, y_cum)
    # end if

    # predict testset
    y_pred_tr = ml_mdl.predict(X_tr_scl)
    y_pred_te = ml_mdl.predict(X_te_scl)

    # extract prediction results
    if prob_type == 'CLF':
        res_pred_tr_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_tr, y_tr)] 
        res_pred_te_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
    elif prob_type == 'REG':
        res_pred_tr_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_tr, y_tr)] 
        res_pred_te_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
    # end if

    return y_pred_tr, y_pred_te, res_pred_tr_idx, res_pred_te_idx