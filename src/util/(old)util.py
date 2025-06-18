##################################################################################################################################################################
# PROJECT: TM_MLOps
# CHAPTER: Util
# SECTION: train
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
#from src.common.config import *
import time

##################################################################################################################################################################
# user-defined utility functions
##################################################################################################################################################################

def run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl):
    # scale dataset
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    # end if

    # train the model
    ml_mdl.fit(X_tr, y_tr)

    # predict testset 
    y_pred_te = ml_mdl.predict(X_te)

    return ml_mdl, y_pred_te

def calculate_attention_matrix(X):
    """
    Compute attention matrix from scaled input window
    """
    scl = StandardScaler()
    X_scl = scl.fit_transform(X)

    attn_scr = np.dot(X_scl.T, X_scl) / np.sqrt(X_scl.shape[0])
    attn_mat = np.exp(attn_scr - np.max(attn_scr, axis=1, keepdims=True))
    attn_mat /= np.sum(attn_mat, axis=1, keepdims=True)

    return attn_mat

def estimate_confidence_interval(ml_mdl, X_tr, y_tr, sqr, conf_lvl):
    """
    Estimate conf. interval based on training results
    Note) 신뢰 구간: 추출된 표본 통계량 기반으로 parameter가 모집단에 포함될 것으로 추정되는 범위

    :param ml_mdl:      trained machine learning model
    :param X_tr:        input of the training set
    :param y_tr:        target of the training set
    :param sqr:         squared value (> 1, type: int)
    :param conf_lvl:    conf. level for error margin (0 ~ 1, type: float)
    :return low_bound:  lower bound of the conf. interval
    :return up_bound:   upper bound of the conf. interval
    """
    # compute residuals
    y_pred    = ml_mdl.predict(X_tr)
    resid     = np.abs(y_tr - y_pred) ** sqr
    len_resid = len(resid)
    
    # compute mean and standard error of residuals
    meanresid      = np.mean(resid)
    std_err_resid   = np.std(resid, ddof = 1) / np.sqrt(len_resid)

    # compute t-value using confidence level (assuming 2-sided test)
    dof     = len_resid - 1  # degree of freedom
    t_value = stats.t.ppf((1 + conf_lvl) / 2, dof)

    # compute lower/upper bound of confidence interval
    mgn_err     = t_value * std_err_resid # 추정치 주변에 허용 가능한 오차 범위
    low_bound   = meanresid - mgn_err
    up_bound    = meanresid + mgn_err

    return low_bound, up_bound

def convert_continuous_to_categorical(y, low_bound, up_bound):
    """
    Convert continuous y values to categorical (binary) based on given bounds.
    :param y:          array-like, target values
    :param low_bound:  lower bound of the confidence interval
    :param up_bound:   upper bound of the confidence interval
    :return:           array-like, categorical (binary) target values
    """

    label = np.where((y >= low_bound) & (y <= up_bound), 1, 0)

    return label

### 241217 calculate CTQ (for posco data)
def hit_ratio(y_true, y_pred, std):
    hit = abs(np.array(y_true) - np.array(y_pred)) < std

    return hit.sum()/len(hit) # hit.sum()/len(hit)*100

### 241217 calculate MAPE (for posco data)
def meanabsolute_percentage_error(y_test, y_pred):
    return np.mean(np.abs((np.array(y_test) - np.array(y_pred))/np.array(y_test)))*100

def split_dataset(X, y, start_idx, end_idx):
    """
    Split dataset into training/test set

    :param  X:          input
    :param  y:          target
    :param  start_idx:  starting index of training set
    :param  end_idx:    end index of traning set
    :return X_tr, X_te, y_tr, y_te: training/test set
    """
    # divide dataset into training/test set
    X_tr, X_te = X.iloc[start_idx:end_idx, :], X.iloc[end_idx:, :] 
    y_tr, y_te = y[start_idx:end_idx], y[end_idx:] 

    return X_tr, X_te, y_tr, y_te
