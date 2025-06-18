##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Algorithm
# SECTION: Concept Drift Detection
# SUB-SECTION: Model Performance-based
# SUB-SUBSECTION: Single Window-based
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import pandas as pd
import numpy as np
import math
import os

from src.util.train import set_ml_dataset, run_ml_model
from src.util.preprocess import OnlineStandardScaler
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ks_2samp, dirichlet, norm, mannwhitneyu, fisher_exact
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
#from src.util.util import run_ml_model_pipeline, calculate_attention_matrix
from datetime import datetime, timedelta

import matplotlib.pyplot as pipet
import matplotlib.patches as patches
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
# Multiple Window-based Methods
##################################################################################################################################################################

##################################################################################################################################################################
# STEPD (Statistical Test of Equal Proportions Difference, 2007) 
##################################################################################################################################################################

class STEPD:
    def __init__(self, **kwargs):
        """
        Initialize values
         **kwargs: {alpha_w: float, alpha_d: float, len_sw: int}

        :param alpha_w:     warning sig.level   (rng: 0 ~ 1)
        :param alpha_d:     drift sig.level     (rng: 0 ~ 1)
        :param len_sw:      length of short window for calculating the recent accuracy (rng: >= 1) 
        """
        # hyperparameters of STEPD
        self.alpha_w  = kwargs['alpha_w']
        self.alpha_d  = kwargs['alpha_d'] 
        self.len_sw   = kwargs['len_sw'] 

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.warn_prd     = []
        self.res_pred_tmp = []

        # cumulative data points, X_cum and y_cum
        # Note) They are used to train the ml model, when the model does not support partial_fit
        self.X_cum = None
        self.y_cum = None

        # dict containing results of cdda
        self.res_cdda = {
            'time_idx':      [], # data index
            'y_real_list':   [], # real targets
            'y_pred_list':   [], # predictions
            'res_pred_list': [], # prediction results (= 0 or 1)
            'cd_idx':        [], # index of concept drfit
            'len_adapt':     [], # length of adaptation period
        }

    @staticmethod
    def _detect_drift(res_pred_lw, res_pred_sw, alpha_w, alpha_d):
        """
        Evaluate the state of the concept

        :param res_pred_lw:   window of prediction results for long window till idx-th point    (type: list)
        :param res_pred_sw:   window of prediction results for short window till idx-th point   (type: list)
        :param alpha_w:       warning sig.level   (rng: 0 ~ 1)    (type: float)
        :param alpha_d:       drift sig.level     (rng: 0 ~ 1)    (type: float)
        :return:              state of the concept
        """
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        # compute p_hat (estimated proportion)
        n_o  = len(res_pred_lw)      # length of long window 
        n_r  = len(res_pred_sw)      # length of short window
        r_o  = np.sum(res_pred_lw)   # num. of corrections among the long window
        r_r  = np.sum(res_pred_sw)   # num. of corrections among the short window
        p_hat   = (r_o + r_r)/(n_o + n_r)  # estimated proportion

        # compute test statistics
        numer      = np.abs(r_o/n_o - r_r/n_r) - 0.5*(1/n_o + 1/n_r)
        denom      = np.sqrt(p_hat * (1 - p_hat) * (1/n_o + 1/n_r))
        test_stat  = numer/(denom + 1e-9)
        
        # compute two-tailed p-value
        p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))

        # evaluate state
        state = 0 if p_val >= alpha_w else 1 if p_val >= alpha_d else 2

        return state

    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_start_idx, te_end_idx):
        """
        Adjust the training set for ml model update
        
        :param state:           state of the concept                 (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param tr_start_idx:    previous start index of training set (type: int)
        :param tr_end_idx:      previous end index of training set   (type: int)        
        :param te_end_idx:      previous end index of test set       (type: int)                
        :return:                updated start index of training set  (type: int)
        """
        if state == 0:  # stable
            tr_start_idx = te_start_idx
            tr_end_idx   = te_end_idx
        elif state == 1:  # warning
            # increment adaptation period
            self.warn_prd.append(te_start_idx) 
            #self.warn_prd.append(te_end_idx) 

            tr_start_idx = te_start_idx
            tr_end_idx   = te_end_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')    

            # increment adaptation period
            self.warn_prd.append(te_end_idx)

            # set the start index of model update
            if (te_end_idx - self.warn_prd[0]) < min_len_tr:
                tr_start_idx = te_end_idx - min_len_tr
            else: 
                tr_start_idx = self.warn_prd[0]
            # end if
            
            tr_end_idx = te_end_idx

            # set the results of cdda
            self.res_cdda['cd_idx'].append(te_end_idx)
            self.res_cdda['len_adapt'].append(tr_end_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.state        = 0
            self.X_cum        = None
            self.y_cum        = None
            self.warn_prd     = []
            self.res_pred_tmp = []
        # end if

        return tr_start_idx, tr_end_idx
    
    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        No parameters to be reset in STEPD
        """
        return None
    
    def run_cdda(self, X, y, scaler, prob_type, ml_mdl, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd):
        """
        Run Concept Drift Detection and Adaptation

        :param X:               input       (type: pd.DataFrame)
        :param y:               target      (type: np.array)
        :param scaler           scaler      (type: TransformerMixin)
        :param prob_type:       problem type (clf: classification / reg: regression) (type: str)
        :param ml_mdl:          ml model   (type: str)
        :param tr_start_idx:    initial start index of training set (type: int)
        :param tr_end_idx:      initial end index of training set   (type: int)        
        :param len_batch        length of batch                     (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param perf_bnd         performance bound for treating prediction results as (in)correct (type: float)
        :return
        """
        # run process
        num_data = len(X)
        while tr_end_idx < num_data:
            # set test set index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, len(X))

            print(f'tr_start_idx: {tr_start_idx} / tr_end_idx: {tr_end_idx} / te_start_idx: {te_start_idx} / te_end_idx: {te_end_idx}')

            # set dataset for ml model
            X_tr, y_tr, X_te, y_te = set_ml_dataset(tr_start_idx = tr_start_idx, 
                                                    tr_end_idx   = tr_end_idx, 
                                                    te_start_idx = te_start_idx, 
                                                    te_end_idx   = te_end_idx,
                                                    X            = X, 
                                                    y            = y)
            # cumulate incoming data points
            self.X_cum = np.concatenate([self.X_cum, X_tr]) if self.X_cum is not None else X_tr
            self.y_cum = np.concatenate([self.y_cum, y_tr]) if self.y_cum is not None else y_tr

            # train the ml model and predict the test set
            y_pred_tr, y_pred_te, res_pred_tr_idx, res_pred_te_idx = run_ml_model(X_cum     = self.X_cum, 
                                                                                  y_cum     = self.y_cum, 
                                                                                  X_tr      = X_tr, 
                                                                                  y_tr      = y_tr, 
                                                                                  X_te      = X_te, 
                                                                                  y_te      = y_te,
                                                                                  y         = y,
                                                                                  scaler    = scaler, 
                                                                                  ml_mdl    = ml_mdl, 
                                                                                  prob_type = prob_type, 
                                                                                  perf_bnd  = perf_bnd)

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(X_te.index)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_te_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_te_idx)

            if len(self.res_pred_tmp) >= 2*self.len_sw:
                # set windows of older/recent prediction results
                res_pred_lw = self.res_pred_tmp[:-self.len_sw]
                res_pred_sw = self.res_pred_tmp[-self.len_sw:]

                print(f'res_pred_lw: {len(res_pred_lw)} / res_pred_sw: {len(res_pred_sw)}')

                # evaluate state of the concept
                self.state = self._detect_drift(res_pred_lw = res_pred_lw, # prediction result for lw 
                                                res_pred_sw = res_pred_sw, # prediction result for sw
                                                alpha_w     = self.alpha_w,
                                                alpha_d     = self.alpha_d)

            # end if

            # set the start/end index of updated training set
            tr_start_idx, tr_end_idx = self._adapt_drift(state        = self.state, 
                                                         min_len_tr   = min_len_tr,
                                                         tr_start_idx = tr_start_idx, 
                                                         tr_end_idx   = tr_end_idx,
                                                         te_start_idx = te_start_idx, 
                                                         te_end_idx   = te_end_idx)
        # end while

        return None

##################################################################################################################################################################
# FHDDMS (Stacking Fast Hoeffding Drift Detection Method, 2018) 
##################################################################################################################################################################

class FHDDMS:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {delta: float, len_lw: int, len_sw: int}

        :param delta:   prob. of m_idx and m_max being different by at least eps  (def: 10e-7, rng: > 0) (type: float)
        :param len_lw:  size of the long window containing prediction results     (def.: 100, rng: >= 1) (type: int)
        :param len_sw:  size of the short window containing prediction results    (def.: 25,  rng: >= 1) (type: int)
        """
        # values to be reset after drift detection
        self.p_max_lw = 0 # max. acc. of long window observed so far
        self.p_max_sw = 0 # max. acc. of short window observed so far

        # hyperparameters of FHDDMs
        self.delta   = kwargs['delta']
        self.len_lw  = kwargs['len_lw'] 
        self.len_sw  = kwargs['len_sw']

        # state / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.res_pred_tmp = []

        # cumulative data points, X_cum and y_cum
        # Note) They are used to train the ml model, when the model does not support partial_fit
        self.X_cum = None
        self.y_cum = None

        # dict containing results of cdda
        self.res_cdda = {
            'time_idx':      [], # data index
            'y_real_list':   [], # real targets
            'y_pred_list':   [], # predictions
            'res_pred_list': [], # prediction results (= 0 or 1)
            'cd_idx':        [], # index of concept drfit
            'len_adapt':     [], # length of adaptation period
        }

    @staticmethod
    def _detect_drift(p_idx_lw, p_max_lw, p_idx_sw, p_max_sw, len_lw, len_sw, delta):
        """
        Evaluate the state of the concept

        :param p_idx_lw:    acc. of theML model for long window                         (type: float)
        :param p_max_lw:    max. acc. of the ML model for long window observed so far   (type: float)
        :param p_idx_sw:    acc. of the ML model for short window                       (type: float)
        :param p_max_sw:    max. acc. of the ML model for short window observed so far  (type: float)
        :param len_lw:      size of the long window containing prediction results     (rng: >= 1) (type: float)
        :param len_sw:      size of the short window containing prediction results    (rng: >= 1) (type: float)
        :param delta:       prob. of m_idx and m_max being different by at least eps  (rng: > 0)  (type: float)
        :return:            state of the concept
        """
        state = 0 # 0: stable / 2: drift

        # compute test statistics and tolerance for drift detection
        test_stat_lw = p_max_lw - p_idx_lw
        test_stat_sw = p_max_sw - p_idx_sw

        eps_drift_lw = np.sqrt(1/(2*len_lw) * math.log(1/delta))
        eps_drift_sw = np.sqrt(1/(2*len_sw) * math.log(1/delta))

        # evaluate state
        state = 2 if (test_stat_lw >= eps_drift_lw) or (test_stat_sw >= eps_drift_sw) else 0

        return state
    
    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_start_idx, te_end_idx):
        """
        Adjust the training set for ml model update
        
        :param state:           state of the concept                 (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param tr_start_idx:    previous start index of training set (type: int)
        :param tr_end_idx:      previous end index of training set   (type: int)        
        :param te_end_idx:      previous end index of test set       (type: int)                
        :return:                updated start index of training set  (type: int)
        """
        if state == 0:  # stable
            tr_start_idx = te_start_idx
            tr_end_idx   = te_end_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set the start index of model update
            tr_start_idx = te_end_idx - min_len_tr 
            tr_end_idx   = te_end_idx

            # set the results of cdda
            self.res_cdda['cd_idx'].append(te_end_idx)
            self.res_cdda['len_adapt'].append(te_end_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.state        = 0
            self.X_cum        = None
            self.y_cum        = None
            self.res_pred_tmp = []
        # end if

        return tr_start_idx, tr_end_idx
    
    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_max_lw = 0
        self.p_max_sw = 0

        return None
    
    def run_cdda(self, X, y, scaler, prob_type, ml_mdl, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd):
        """
        Run Concept Drift Detection and Adaptation

        :param X:               input       (type: pd.DataFrame)
        :param y:               target      (type: np.array)
        :param scaler           scaler      (type: TransformerMixin)
        :param prob_type:       problem type (clf: classification / reg: regression) (type: str)
        :param ml_mdl:          ml model   (type: str)
        :param tr_start_idx:    initial start index of training set (type: int)
        :param tr_end_idx:      initial end index of training set   (type: int)        
        :param len_batch        length of batch                     (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param perf_bnd         performance bound for treating prediction results as (in)correct (type: float)
        :return
        """
        # run process
        num_data = len(X)
        while tr_end_idx < num_data:
            # set test set index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, len(X))

            print(f'tr_start_idx: {tr_start_idx} / tr_end_idx: {tr_end_idx} / te_start_idx: {te_start_idx} / te_end_idx: {te_end_idx}')

            # set dataset for ml model
            X_tr, y_tr, X_te, y_te = set_ml_dataset(tr_start_idx = tr_start_idx, 
                                                    tr_end_idx   = tr_end_idx, 
                                                    te_start_idx = te_start_idx, 
                                                    te_end_idx   = te_end_idx,
                                                    X            = X, 
                                                    y            = y)
            # cumulate incoming data points
            self.X_cum = np.concatenate([self.X_cum, X_tr]) if self.X_cum is not None else X_tr
            self.y_cum = np.concatenate([self.y_cum, y_tr]) if self.y_cum is not None else y_tr

            # train the ml model and predict the test set
            y_pred_tr, y_pred_te, res_pred_tr_idx, res_pred_te_idx = run_ml_model(X_cum     = self.X_cum, 
                                                                                  y_cum     = self.y_cum, 
                                                                                  X_tr      = X_tr, 
                                                                                  y_tr      = y_tr, 
                                                                                  X_te      = X_te, 
                                                                                  y_te      = y_te,
                                                                                  y         = y,
                                                                                  scaler    = scaler, 
                                                                                  ml_mdl    = ml_mdl, 
                                                                                  prob_type = prob_type, 
                                                                                  perf_bnd  = perf_bnd)

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(X_te.index)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_te_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_te_idx)

            if len(self.res_pred_tmp) >= self.len_lw:
                # drop and push element
                res_pred_lw = self.res_pred_tmp
                res_pred_sw = res_pred_lw[-self.len_sw:] 
              
                # compute the accuracy for long/short window
                p_idx_lw = sum(res_pred_lw)/len(res_pred_lw)
                p_idx_sw = sum(res_pred_sw)/len(res_pred_sw)
            
                # update p_max if p_max < p_idx
                self.p_max_lw = p_idx_lw if self.p_max_lw < p_idx_lw else self.p_max_lw
                self.p_max_sw = p_idx_sw if self.p_max_sw < p_idx_sw else self.p_max_sw

                # evaluate state
                self.state = self._detect_drift(p_idx_lw  = p_idx_lw, 
                                                p_max_lw  = self.p_max_lw, 
                                                p_idx_sw  = p_idx_sw, 
                                                p_max_sw  = self.p_max_sw, 
                                                len_lw    = self.len_lw,
                                                len_sw    = self.len_sw,
                                                delta     = self.delta)

                # drop entry into sliding window
                self.res_pred_tmp = self.res_pred_tmp[len_batch:]             
            # end if

            # set the start/end index of updated training set
            tr_start_idx, tr_end_idx = self._adapt_drift(state        = self.state, 
                                                         min_len_tr   = min_len_tr,
                                                         tr_start_idx = tr_start_idx, 
                                                         tr_end_idx   = tr_end_idx,
                                                         te_start_idx = te_start_idx, 
                                                         te_end_idx   = te_end_idx)
        # end while

        return None

##################################################################################################################################################################
# MR-DDM (Multi-Resolution Drift Detection Method, 2025) 
##################################################################################################################################################################

class MRDDM:
    def __init__(self, **kwargs):
        # hyperparameters of MRDDM
        self.alpha_d  = kwargs['alpha_d']
        self.len_step = kwargs['len_step'] # length of step
        self.len_sw   = kwargs['len_sw'] 

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        #self.warn_prd     = []
        self.res_pred_tmp = []

        self.res_pred_tr_tmp = []

        # cumulative data points, X_cum and y_cum
        # Note) They are used to train the ml model, when the model does not support partial_fit
        self.X_cum = None
        self.y_cum = None

        # dict containing results of cdda
        self.res_cdda = {
            'time_idx':      [], # data index
            'y_real_list':   [], # real targets
            'y_pred_list':   [], # predictions
            'res_pred_list': [], # prediction results (= 0 or 1)
            'cd_idx':        [], # index of concept drfit
            'len_adapt':     [], # length of adaptation period
        }

    """
    @staticmethod
    def _detect_drift(res_pred_lw, res_pred_sw, alpha_d):
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        num_obs    = len(res_pred_lw)
        num_lw_cor = sum(res_pred_lw)
        num_sw_cor = sum(res_pred_sw)

        #num_cor_arr = np.array([num_lw_cor, num_sw_cor])
        #num_obs_arr = np.array([num_obs, num_obs])

        #_, p_val = proportions_ztest(num_cor_arr, num_obs_arr, alternative='two-sided')

        # [[success_X, failure_X], [success_Y, failure_Y]]
        res_table = [[num_lw_cor, num_obs-num_lw_cor], 
                     [num_sw_cor, num_obs-num_sw_cor]]

        stat, p_val = fisher_exact(res_table, alternative='two-sided')

        print('MRDDM')
        print(f'num_obs: {num_obs} / num_lw_cor: {num_lw_cor} / num_sw_cor: {num_sw_cor} / p-val: {p_val}')

        state = 2 if p_val < alpha_d else 0

        return state
    """

    @staticmethod
    def _detect_drift(res_pred_lw, res_pred_sw, res_pred_tr_tmp, alpha_d):
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        num_tr_obs = len(res_pred_tr_tmp)
        num_tr_cor = sum(res_pred_tr_tmp)
        num_te_obs = len(res_pred_lw)
        num_lw_cor = sum(res_pred_lw)
        num_sw_cor = sum(res_pred_sw)

        # ztest
        num_cor_arr_lw = np.array([num_tr_cor, num_lw_cor])
        num_obs_arr_lw = np.array([num_tr_obs, num_te_obs])

        num_cor_arr_sw = np.array([num_tr_cor, num_sw_cor])
        num_obs_arr_sw = np.array([num_tr_obs, num_te_obs])

        _, p_val_lw = proportions_ztest(num_cor_arr_lw, num_obs_arr_lw, alternative='two-sided')
        _, p_val_sw = proportions_ztest(num_cor_arr_sw, num_obs_arr_sw, alternative='two-sided')

        # state: 0 -> stable / 1 -> lw cd / 2 -> sw cd / 3 -> lw/sw cd
        state = (
                0 if (p_val_lw  > alpha_d) and (p_val_sw  > alpha_d) else
                1 if (p_val_lw <= alpha_d) and (p_val_sw  > alpha_d) else 
                2 if (p_val_lw  > alpha_d) and (p_val_sw <= alpha_d) else 
                3
        )

        ## fisher exact test
        # #[[success_X, failure_X], [success_Y, failure_Y]]
        # res_tab_lw = [[num_tr_cor, num_tr_obs - num_tr_cor], 
        #               [num_lw_cor, num_te_obs - num_lw_cor]]

        # res_tab_sw = [[num_tr_cor, num_tr_obs - num_tr_cor], 
        #               [num_sw_cor, num_te_obs - num_sw_cor]]
        
        # _, p_val_lw = fisher_exact(res_tab_lw, alternative='two-sided')
        # _, p_val_sw = fisher_exact(res_tab_sw, alternative='two-sided')


        # # state: 0 -> stable / 1 -> lw cd / 2 -> sw cd / 3 -> lw/sw cd
        # state = (
        #         0 if (p_val_lw  > alpha_d) and (p_val_sw  > alpha_d) else
        #         1 if (p_val_lw <= alpha_d) and (p_val_sw  > alpha_d) else 
        #         2 if (p_val_lw  > alpha_d) and (p_val_sw <= alpha_d) else 
        #         3
        # )

        return state
    
    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_start_idx, te_end_idx, len_step, len_sw):
        if state == 0 or state == 3:  # stable or whole drift
            tr_start_idx = te_start_idx
            tr_end_idx   = te_end_idx
        elif state == 1 or state == 2 : # long window drift or short window drift
            print(f'Drift detected at {te_end_idx}')  
            if state == 1:
                ## set drift index
                #drift_idx = te_end_idx - len_step*len_sw
                tr_start_idx = te_end_idx - len_step*len_sw
            elif state == 2:
                # set drift index
                #drift_idx = te_end_idx - len_sw
                tr_start_idx = te_end_idx - len_sw
            # end if

            # tr_start_idx = drift_idx - min_len_tr
            tr_end_idx   = te_end_idx

            # print(f'state: {state}')
            # print(f'len_adpat: {tr_end_idx-tr_start_idx}')

            # set the results of cdda
            self.res_cdda['cd_idx'].append(te_end_idx)
            self.res_cdda['len_adapt'].append(tr_end_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.state        = 0
            self.X_cum        = None
            self.y_cum        = None
            self.res_pred_tmp = []
            self.res_pred_tr_tmp = []
        # end if

        return tr_start_idx, tr_end_idx
    
    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        No parameters to be reset in MRDDM
        """
        return None
    
    def run_cdda(self, X, y, scaler, prob_type, ml_mdl, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd):
        """
        Run Concept Drift Detection and Adaptation

        :param X:               input       (type: np.array)
        :param y:               target      (type: np.array)
        :param scaler           scaler      (type: TransformerMixin)
        :param prob_type:       problem type (clf: classification / reg: regression) (type: str)
        :param ml_mdl:          ml model   (type: str)
        :param tr_start_idx:    initial start index of training set (type: int)
        :param tr_end_idx:      initial end index of training set   (type: int)        
        :param len_batch        length of batch                     (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param perf_bnd         performance bound for treating prediction results as (in)correct (type: float)
        :return
        """
        # run process
        num_data = len(X)
        while tr_end_idx < num_data:
            # set test set index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, len(X))

            print(f'tr_start_idx: {tr_start_idx} / tr_end_idx: {tr_end_idx} / te_start_idx: {te_start_idx} / te_end_idx: {te_end_idx}')

            # set dataset for ml model
            X_tr, y_tr, X_te, y_te = set_ml_dataset(tr_start_idx = tr_start_idx, 
                                                    tr_end_idx   = tr_end_idx, 
                                                    te_start_idx = te_start_idx, 
                                                    te_end_idx   = te_end_idx,
                                                    X            = X, 
                                                    y            = y)
            # cumulate incoming data points
            self.X_cum = np.concatenate([self.X_cum, X_tr]) if self.X_cum is not None else X_tr
            self.y_cum = np.concatenate([self.y_cum, y_tr]) if self.y_cum is not None else y_tr

            # train the ml model and predict the test set
            y_pred_tr, y_pred_te, res_pred_tr_idx, res_pred_te_idx = run_ml_model(X_cum     = self.X_cum, 
                                                                                  y_cum     = self.y_cum, 
                                                                                  X_tr      = X_tr, 
                                                                                  y_tr      = y_tr, 
                                                                                  X_te      = X_te, 
                                                                                  y_te      = y_te,
                                                                                  y         = y,
                                                                                  scaler    = scaler, 
                                                                                  ml_mdl    = ml_mdl, 
                                                                                  prob_type = prob_type, 
                                                                                  perf_bnd  = perf_bnd)

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(X_te.index)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_te_idx)

            ###
            self.res_pred_tr_tmp.extend(res_pred_tr_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_te_idx)

            if len(self.res_pred_tmp) >= self.len_step*self.len_sw:
                # set windows of older/recent prediction results
                res_pred_lw = self.res_pred_tmp[::self.len_step]                
                res_pred_sw = self.res_pred_tmp[-self.len_sw:]

                # print(f'len(self.res_pred_tr_tmp): {len(self.res_pred_tr_tmp)}')
                # print(f'len(self.res_pred_tmp): {len(self.res_pred_tmp)}')
                # print(f'len(res_pred_lw): {len(res_pred_lw)}')
                # print(f'len(res_pred_lw): {len(res_pred_sw)}')

                #self.state = self._detect_drift(res_pred_lw = res_pred_lw, 
                #                                res_pred_sw = res_pred_sw, 
                #                                alpha_d     = self.alpha_d)
                
                self.state = self._detect_drift(res_pred_lw     = res_pred_lw, 
                                                res_pred_sw     = res_pred_sw, 
                                                res_pred_tr_tmp = self.res_pred_tr_tmp,
                                                alpha_d         = self.alpha_d)
                
                # drop entry into sliding window
                self.res_pred_tmp = self.res_pred_tmp[len_batch:]
            # end if

            # set the start/end index of updated training set
            tr_start_idx, tr_end_idx = self._adapt_drift(state        = self.state, 
                                                         min_len_tr   = min_len_tr,
                                                         tr_start_idx = tr_start_idx, 
                                                         tr_end_idx   = tr_end_idx,
                                                         te_start_idx = te_start_idx, 
                                                         te_end_idx   = te_end_idx,
                                                         len_step     = self.len_step,
                                                         len_sw       = self.len_sw)
        # end while

        return None