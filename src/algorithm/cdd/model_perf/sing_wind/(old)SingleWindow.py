##################################################################################################################################################################
# PROJECT: TM_MLOps
# CHAPTER: Algorithm
# SECTION: Concept Drift Detection
# SUB-SECTION: Model Performance-based
# SUB-SUBSECTION: Single Window-based
# AUTHOR: Yang et al.
# DATE: since 24.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import pandas as pd
import numpy as np
import math
import os

from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
from scipy.stats import ks_2samp, dirichlet, norm, mannwhitneyu
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
# Single Window-based Methods
##################################################################################################################################################################

##################################################################################################################################################################
# DDM_ATTN (Drift Detection Method with Attention, 2025) 
##################################################################################################################################################################

class DDM_ATTN:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {alpha_w: float, alpha_d: float, clt_idx: int}

        :param alpha_w:     warning conf.level                    (rng: >= 1)   (type: float)            
        :param alpha_d:     drift conf.level                      (rng: >= 1)   (type: float)
        :param clt_idx:     min. num. of data points to obey CLT  (rng: >= 30)  (type: int)
        """
        # values to be reset after drift detection
        self.p_min = np.inf   # min. err. rate of the ML model
        self.s_min = np.inf   # min. std. dev. of the ML model

        # hyperparameters of DDM
        self.alpha_w = kwargs['alpha_w']
        self.alpha_d = kwargs['alpha_d'] 
        self.clt_idx = kwargs['clt_idx'] 
        self.top_n   = kwargs['top_n']

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.warn_prd     = []
        self.res_pred_tmp = []

        # TODO
        self.vd_prd = []

        # dict containing results of cdda
        self.res_cdda = {
            'time_idx':      [], # data index
            'y_real_list':   [], # real targets
            'y_pred_list':   [], # predictions
            'res_pred_list': [], # prediction results (= 0 or 1)
            'cd_idx':        [], # index of concept drfit
            'len_adapt':     [], # length of adaptation period
        }

    # TODO: step size: len_batch?
    @staticmethod
    def _build_virtual_drift_period(X, tr_start_idx, tr_end_idx, clt_idx, top_n):
        """
        Generate dictionary of attention matrices from X[tr_start_idx:tr_end_idx]
        using sliding window of length clt_idx

        :param X:            input data (type: pd.DataFrame or np.ndarray)
        :param tr_start_idx: starting index (inclusive)
        :param tr_end_idx:   ending index (exclusive)
        :param clt_idx:      window length
        :return:             dict contatining attention matrices
        """

        attn_mat_dict = {}
        # TODO: step_size
        for idx in range(tr_start_idx, tr_end_idx, 12):
            start_idx = idx
            end_idx   = min(idx + clt_idx, tr_end_idx)            
            X_idx     = X[start_idx:end_idx]
            attn_mat  = calculate_attention_matrix(X=X_idx)
            
            key = str(start_idx) + '_' + str(end_idx)
            attn_mat_dict[key] = attn_mat
        # end for
        
        """
        # Reference matrix (last window)
        ref_key = list(attn_mat_dict.keys())[-1]
        attn_mat_ref = attn_mat_dict[ref_key].flatten()
        attn_mat_ref /= attn_mat_ref.sum()  # Normalize for JSD

        # Compute JSD from ref to others
        jsd_list = []
        for key, attn_mat in attn_mat_dict.items():
            if key == ref_key:
                continue
            attn_vec = attn_mat.flatten()
            attn_vec /= attn_vec.sum()  # Normalize for JSD

            jsd = jensenshannon(attn_mat_ref, attn_vec)
            jsd_list.append((key, jsd))

        # Sort by JSD (ascending = more similar)
        jsd_list.sort(key=lambda x: x[1])
        top_n_keys = [key for key, _ in jsd_list[:top_n]]

        # Build virtual drift period
        vd_prd = []
        for key in top_n_keys:
            start_idx, end_idx = map(int, key.split('_'))
            vd_prd.extend(list(range(start_idx, end_idx)))

        vd_prd = list(set(sorted(vd_prd)))

        return vd_prd
        """
        # get reference attention matrix (last one)
        ref_key = list(attn_mat_dict.keys())[-1]
        attn_mat_ref = attn_mat_dict[ref_key]

        # compute frobeniusm norm between reference and other matrices
        frob_norm_list = []
        for key, attn_mat_idx in attn_mat_dict.items():
            if key == ref_key:
                continue
            frob_norm = np.linalg.norm(attn_mat_ref - attn_mat_idx, ord='fro')
            frob_norm_list.append((key, frob_norm))
        # end for

        # sort by frobeniusm norm (ascending)
        frob_norm_list.sort(key=lambda x: x[1])

        # pick top_n keys
        top_n_keys = [key for key, _ in frob_norm_list[:top_n]]

        # convert keys to index ranges
        vd_prd = []
        for key in top_n_keys:
            start_idx, end_idx = map(int, key.split('_'))
            vd_prd.extend(list(range(start_idx, end_idx)))
        # end for

        vd_prd = list(set(sorted(vd_prd)))

        return vd_prd
        
    @staticmethod
    def _detect_drift(p_idx, s_idx, p_min, s_min, alpha_w, alpha_d):
        """
        Evaluate the state of the concept

        :param p_idx:       err. rate of ML model untill idx-th point   (type: float)  
        :param s_idx        std. dev. of ML model untill idx-th point   (type: float)
        :param p_min:       min. err. rate of ML model observed so far  (type: float)
        :param s_min        min. std. dev. of ML model observed so far  (type: float)
        :param alpha_w:     warning conf.level     (rng: >= 1)          (type: float)
        :param alpha_d:     drift conf.level       (rng: >= 1)          (type: float)
        :return:            state of the concept
        """
        state = 0 # 0: stable, 1: warning, 2: drift

        # compute test statistics for drift detection
        test_stat = p_idx + s_idx

        # evaluate state
        ### 250109 DHY: if 문에 >= 대신 > 사용
        ### 모든 예측이 정답(p_idx와 s_idx 모두 0)이면 p_min과 s_min도 0이됨 => state는 2가 되고 cd가 컨펌됨 
        ### 모든 예측이 정답인데 cd가 컨펌되서는 안되므로 코드상에서는 > 를 사용하고자 함        
        state = 2 if test_stat > (p_min + alpha_d * s_min) else \
                1 if test_stat > (p_min + alpha_w * s_min) else \
                0

        return state

    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
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
            tr_start_idx = tr_start_idx
        elif state == 1:  # warning
            # increment adaptation period
            self.warn_prd.append(te_end_idx)
            tr_start_idx = tr_start_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set drift index
            drift_idx = te_end_idx

            # increment adaptation period
            self.warn_prd.append(drift_idx) 

            # set the start index of model update
            if (drift_idx - self.warn_prd[0]) <= min_len_tr:
                tr_start_idx = drift_idx - min_len_tr 
            else: 
                tr_start_idx = self.warn_prd[0]
            # end if

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.warn_prd     = []
            self.res_pred_tmp = []
        # end if

        return tr_start_idx

    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_min    = np.inf
        self.s_min    = np.inf

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
            # set test set / time index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)
            te_time_idx  = X.iloc[te_start_idx:te_end_idx].index

            tr_idx_list = list(range(tr_start_idx, tr_end_idx)) + list(self.vd_prd)
            te_idx_list = list(range(te_start_idx, te_end_idx))

            # create training/test set
            print(len(tr_idx_list))
            X_tr, y_tr = X.iloc[tr_idx_list], y[tr_idx_list]
            X_te, y_te = X.iloc[te_idx_list], y[te_idx_list]

            # train ml model and predict testset
            ml_mdl, y_pred_te = run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl)

            # extract prediction results
            if prob_type == 'CLF':
                res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
            elif prob_type == 'REG':
                res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
            # end if

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(te_time_idx)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.clt_idx:
                # compute err. rate and std. dev.
                p_idx = 1 - sum(self.res_pred_tmp)/len(self.res_pred_tmp)
                s_idx = np.sqrt(p_idx*(1-p_idx)/len((self.res_pred_tmp)))

                # update p_min and s_min if p_idx + s_idx is lower than p_min + s_min
                self.p_min = p_idx if p_idx + s_idx < self.p_min + self.s_min else self.p_min
                self.s_min = s_idx if p_idx + s_idx < self.p_min + self.s_min else self.s_min

                # evaluate state of the concept
                self.state = self._detect_drift(p_idx   = p_idx,
                                                s_idx   = s_idx, 
                                                p_min   = self.p_min, 
                                                s_min   = self.s_min, 
                                                alpha_w = self.alpha_w, 
                                                alpha_d = self.alpha_d)

                if self.state == 2:
                    self.vd_prd = self._build_virtual_drift_period(X             = X, 
                                                                   tr_start_idx  = tr_start_idx, 
                                                                   tr_end_idx    = tr_end_idx, 
                                                                   clt_idx       = self.clt_idx, 
                                                                   top_n         = self.top_n)
                    print('1111111111111111111111111')
                    print(len(self.vd_prd), len(self.vd_prd)/self.clt_idx, tr_end_idx)

                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 min_len_tr   = min_len_tr,
                                                 tr_start_idx = tr_start_idx, 
                                                 tr_end_idx   = tr_end_idx, 
                                                 te_end_idx   = te_end_idx)
            # end if
            
            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None

##################################################################################################################################################################
# DDM (Drift Detection Method, 2004) 
##################################################################################################################################################################

class DDM:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {alpha_w: float, alpha_d: float, clt_idx: int}

        :param alpha_w:     warning conf.level                    (rng: >= 1)   (type: float)            
        :param alpha_d:     drift conf.level                      (rng: >= 1)   (type: float)
        :param clt_idx:     min. num. of data points to obey CLT  (rng: >= 30)  (type: int)
        """
        # values to be reset after drift detection
        self.p_min = np.inf   # min. err. rate of the ML model
        self.s_min = np.inf   # min. std. dev. of the ML model

        # hyperparameters of DDM
        self.alpha_w = kwargs['alpha_w']
        self.alpha_d = kwargs['alpha_d'] 
        self.clt_idx = kwargs['clt_idx'] 

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.warn_prd     = []
        self.res_pred_tmp = []

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
    def _detect_drift(p_idx, s_idx, p_min, s_min, alpha_w, alpha_d):
        """
        Evaluate the state of the concept

        :param p_idx:       err. rate of ML model untill idx-th point   (type: float)  
        :param s_idx        std. dev. of ML model untill idx-th point   (type: float)
        :param p_min:       min. err. rate of ML model observed so far  (type: float)
        :param s_min        min. std. dev. of ML model observed so far  (type: float)
        :param alpha_w:     warning conf.level     (rng: >= 1)          (type: float)
        :param alpha_d:     drift conf.level       (rng: >= 1)          (type: float)
        :return:            state of the concept
        """
        state = 0 # 0: stable, 1: warning, 2: drift

        # compute test statistics for drift detection
        test_stat = p_idx + s_idx

        # evaluate state
        ### 250109 DHY: if 문에 >= 대신 > 사용
        ### 모든 예측이 정답(p_idx와 s_idx 모두 0)이면 p_min과 s_min도 0이됨 => state는 2가 되고 cd가 컨펌됨 
        ### 모든 예측이 정답인데 cd가 컨펌되서는 안되므로 코드상에서는 > 를 사용하고자 함        
        state = 2 if test_stat > (p_min + alpha_d * s_min) else \
                1 if test_stat > (p_min + alpha_w * s_min) else \
                0

        return state

    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
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
            tr_start_idx = tr_start_idx
        elif state == 1:  # warning
            # increment adaptation period
            self.warn_prd.append(te_end_idx)
            tr_start_idx = tr_start_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set drift index
            drift_idx = te_end_idx

            # increment adaptation period
            self.warn_prd.append(drift_idx) 

            # set the start index of model update
            if (drift_idx - self.warn_prd[0]) <= min_len_tr:
                tr_start_idx = drift_idx - min_len_tr 
            else: 
                tr_start_idx = self.warn_prd[0]
            # end if

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.warn_prd     = []
            self.res_pred_tmp = []
        # end if

        return tr_start_idx

    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_min    = np.inf
        self.s_min    = np.inf

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
            # set test set / time index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)
            te_time_idx  = X.iloc[te_start_idx:te_end_idx].index

            # create training/test set
            X_tr, y_tr = X.iloc[tr_start_idx:tr_end_idx], y[tr_start_idx:tr_end_idx]
            X_te, y_te = X.iloc[te_start_idx:te_end_idx], y[te_start_idx:te_end_idx]

            #print('*' * 500)
            #print(f'state: {self.state}, te_end_idx: {te_end_idx} /{num_data}')
            #print(f'tr_start_idx: {tr_start_idx} / tr_end_idx: {tr_end_idx} / len_tr: {tr_end_idx-tr_start_idx}')
            #print(f'te_start_idx: {te_start_idx} / te_end_idx: {te_end_idx} / len_te: {te_end_idx-te_start_idx}')

            # train ml model and predict testset
            ml_mdl, y_pred_te = run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl)

            # extract prediction results
            if prob_type == 'CLF':
                res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
            elif prob_type == 'REG':
                res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
            # end if

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(te_time_idx)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.clt_idx:
                #print('length of predictions:', len(self.res_pred_tmp))

                # compute err. rate and std. dev.
                p_idx = 1 - sum(self.res_pred_tmp)/len(self.res_pred_tmp)
                s_idx = np.sqrt(p_idx*(1-p_idx)/len((self.res_pred_tmp)))

                # update p_min and s_min if p_idx + s_idx is lower than p_min + s_min
                self.p_min = p_idx if p_idx + s_idx < self.p_min + self.s_min else self.p_min
                self.s_min = s_idx if p_idx + s_idx < self.p_min + self.s_min else self.s_min

                # evaluate state of the concept
                self.state = self._detect_drift(p_idx   = p_idx,
                                                s_idx   = s_idx, 
                                                p_min   = self.p_min, 
                                                s_min   = self.s_min, 
                                                alpha_w = self.alpha_w, 
                                                alpha_d = self.alpha_d)

                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 min_len_tr   = min_len_tr,
                                                 tr_start_idx = tr_start_idx, 
                                                 tr_end_idx   = tr_end_idx, 
                                                 te_end_idx   = te_end_idx)
            # end if
            
            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None

##################################################################################################################################################################
# FHDDM (Fast Hoeffding Drift Detection Method, 2016) 
##################################################################################################################################################################

class FHDDM:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {delta: float, len_wind: int}

        :param delta:       prob. of p_idx and p_max being different by at least eps_drift  (rng: > 0)  (type: float)
        :param len_wind:    length of the sliding window containing prediction results      (rng: >= 1) (type: int)
        """
        # values to be reset after drift detection
        self.p_max = 0   # max. acc. of the ML model observed so far

        # hyperparameters of FHDDM
        self.delta    = kwargs['delta']
        self.len_wind = kwargs['len_wind'] 

        # state / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.res_pred_tmp = []

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
    def _detect_drift(p_idx, p_max, delta, len_wind):
        """
        Evaluate the state of the concept

        :param p_idx:       acc. of the ML model untill idx-th point    (type: float)
        :param p_max:       max. acc. of ML model observed so far       (type: float)
        :param delta:       prob. of p_idx and p_max being different by at least eps_drift  (rng: > 0)  (type: float)
        :param len_wind:    length of the sliding window containing prediction results      (rng: >= 1) (type: int)
        :return:            state of the concept
        """
        state = 0 # 0: stable, 2: drift

        # compute test statistics and tolerance for drift detection
        test_stat = p_max - p_idx
        eps_drift = np.sqrt(1/(2*len_wind) * math.log(1/delta))

        # evaluate state
        state = 2 if test_stat >= eps_drift else 0

        return state
    
    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
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
            tr_start_idx = tr_start_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set drift index
            drift_idx = te_end_idx

            # set the start index of model update
            tr_start_idx = drift_idx - min_len_tr 

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.res_pred_tmp = []
        # end if

        return tr_start_idx
    
    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_max = 0   

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
            # set test set / time index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)
            te_time_idx  = X.iloc[te_start_idx:te_end_idx].index

            # create training/test set
            X_tr, y_tr = X.iloc[tr_start_idx:tr_end_idx], y[tr_start_idx:tr_end_idx]
            X_te, y_te = X.iloc[te_start_idx:te_end_idx], y[te_start_idx:te_end_idx]

            # train ml model and predict testset
            ml_mdl, y_pred_te = run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl)

            # extract prediction results
            if prob_type == 'CLF':
                res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
            elif prob_type == 'REG':
                res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
            # end if

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(te_time_idx)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.len_wind:
                # compute the accuracy till idx-th point
                p_idx = sum(self.res_pred_tmp)/len(self.res_pred_tmp)

                # update p_max if p_max < p_idx
                self.p_max = p_idx if self.p_max < p_idx else self.p_max

                # evaluate state
                self.state = self._detect_drift(p_idx    = p_idx, 
                                                p_max    = self.p_max, 
                                                delta    = self.delta, 
                                                len_wind = self.len_wind)

                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 min_len_tr   = min_len_tr,
                                                 tr_start_idx = tr_start_idx, 
                                                 tr_end_idx   = tr_end_idx, 
                                                 te_end_idx   = te_end_idx)

                # drop entry into sliding window
                self.res_pred_tmp = self.res_pred_tmp[len_batch:]
            # end if

            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None

##################################################################################################################################################################
# MDDM (McDiarmid Drift Detection Methods, 2018) 
##################################################################################################################################################################

class MDDM:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {delta: float, len_wind: int, wgt_type: str, wgt_diff/wgt_rate/wgt_lambda: float}

        :param delta:       prob. of p_idx and p_max being different by at least eps_drift  (rng: > 0)  (type: float)
        :param len_wind:    length of the sliding window containing prediction results      (rng: >= 1) (type: int)
        :param wgt_type:    type of weighthing schemes (a: arithmetic, g: geometric, e: euler)          (type: str)
        :param wgt_diff:    weight difference (for arithmetic scheme)   (type: float)
        :param wgt_rate:    weight rate       (for geometric scheme)    (type: float)
        :param wgt_lambda:  weight rate       (for euler scheme)        (type: float)
        """
        # values to be reset after drift detection
        self.p_max = 0   # max. acc. of the ML model observed so far

        # hyperparameters of MDDM
        self.delta       = kwargs['delta']
        self.len_wind    = kwargs['len_wind'] 
        self.wgt_type    = kwargs['wgt_type']
        self.wgt_diff    = kwargs['wgt_diff']
        self.wgt_rate    = kwargs['wgt_rate']
        self.wgt_lambda  = kwargs['wgt_lambda']

        # state / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.res_pred_tmp = []

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
    def _calculate_entry_weight(len_wind, wgt_type, wgt_diff, wgt_rate, wgt_lambda):
        """
        :param len_wind:    length of the sliding window containing prediction results      (rng: >= 1) (type: int)
        :param wgt_type:    type of weighthing schemes (a: arithmetic, g: geometric, e: euler)          (type: str)
        :param wgt_diff:    weight difference (for arithmetic scheme)   (type: float)
        :param wgt_rate:    weight rate       (for geometric scheme)    (type: float)
        :param wgt_lambda:  weight rate       (for euler scheme)        (type: float)
        :return:            list containing weight for each entry
        """
        if wgt_type == 'a':   # arithmetic (wgt_diff >= 0))
            entry_wgt_list = [1 + (i - 1) * wgt_diff    for i in range(1, len_wind + 1)]
        elif wgt_type == 'g': # geometric (wgt_rate >= 1)
            entry_wgt_list = [np.power(wgt_rate, i - 1) for i in range(1, len_wind + 1)]
        elif wgt_type == 'e': # euler (wgt_lambda >= 0)
            wgt_rate       = np.exp(wgt_lambda) # e^lambda
            entry_wgt_list = [np.power(wgt_rate, i - 1) for i in range(1, len_wind + 1)]
        # end if

        return entry_wgt_list
    
    @staticmethod 
    def _detect_drift(p_idx, p_max, delta, entry_wgt_list):
        """
        Evaluate the state of the concept

        :param p_idx:           acc. of  ML model untill idx-th point   (type: float)
        :param p_max:           max. acc. of ML model observed so far   (type: float)
        :param delta:           prob. of p_idx and p_max being different by at least eps_drift (rng: > 0) (type: float)
        :param entry_wgt_list:  list of weights for each entry in the sliding window    (type: list)
        :return:                state of the concept
        """
        state = 0 # 0: stable, 2: drift

        # compute test statistics
        test_stat = p_max - p_idx

        # compute tolerance for drift detection
        len_wind         = len(entry_wgt_list)
        sum_enty_wgt     = np.sum(entry_wgt_list)
        rel_wgt_list     = [entry_wgt_list[idx] / sum_enty_wgt    for idx in range(len_wind)]  # relative weight
        sum_sqr_rel_wgt  = np.sum([np.power(rel_wgt_list[idx], 2) for idx in range(len_wind)]) # sum of squared relative weight

        eps_drift = np.sqrt(sum_sqr_rel_wgt/2 * math.log(1/delta))  

        # evaluate state
        state = 2 if test_stat >= eps_drift else 0

        return state

    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
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
            tr_start_idx = tr_start_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set drift index
            drift_idx = te_end_idx

            # set the start index of model update
            tr_start_idx = drift_idx - min_len_tr 

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.res_pred_tmp = []
        # end if

        return tr_start_idx
    
    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_max = 0   

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
            # set test set / time index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)
            te_time_idx  = X.iloc[te_start_idx:te_end_idx].index

            # create training/test set
            X_tr, y_tr = X.iloc[tr_start_idx:tr_end_idx], y[tr_start_idx:tr_end_idx]
            X_te, y_te = X.iloc[te_start_idx:te_end_idx], y[te_start_idx:te_end_idx]

            # train ml model and predict testset
            ml_mdl, y_pred_te = run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl)

            # extract prediction results
            if prob_type == 'CLF':
                res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
            elif prob_type == 'REG':
                res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
            # end if

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(te_time_idx)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.len_wind:
                # compute the weight for each entry in sliding window
                entry_wgt_list = self._calculate_entry_weight(len_wind   = self.len_wind, 
                                                              wgt_type   = self.wgt_type, 
                                                              wgt_diff   = self.wgt_diff, 
                                                              wgt_rate   = self.wgt_rate, 
                                                              wgt_lambda = self.wgt_lambda)
                
                # compute the weighted accuracy till idx-th point
                numer = np.sum([entry_wgt_list[idx] * self.res_pred_tmp[idx] for idx in range(self.len_wind)])
                denom = np.sum(entry_wgt_list)
                p_idx = numer/(denom + 1e-9)

                # update p_max if p_max < p_idx
                self.p_max = p_idx if self.p_max < p_idx else self.p_max

                # evaluate state
                self.state = self._detect_drift(p_idx          = p_idx,
                                                p_max          = self.p_max, 
                                                delta          = self.delta, 
                                                entry_wgt_list = entry_wgt_list)
                
                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 min_len_tr   = min_len_tr,
                                                 tr_start_idx = tr_start_idx, 
                                                 tr_end_idx   = tr_end_idx, 
                                                 te_end_idx   = te_end_idx)

                # drop entry into sliding window
                self.res_pred_tmp = self.res_pred_tmp[len_batch:]
            # end if

            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None
    
##################################################################################################################################################################
# BDDM (Bhattacharyya Drift Detection Methods, 2021) 
##################################################################################################################################################################

class BDDM:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {delta: float, len_wind: int, wgt_diff: float}

        :param delta:       conf. value for FPR bound (rng: 0 ~ 1)      (type: float)
        :param len_wind:    length of the sliding window containing prediction results (rng: >= 1) (type: int)
        :param wgt_diff:    weight difference (for arithmetic scheme)   (type: float)
        """
        # values to be reset after drift detection
        self.p_max = 0   # max. acc. of the ML model observed so far
        self.s_max = 0   # std. dev. of the ML model in respect to p_max

        # hyperparameters of BDDM
        self.delta     = kwargs['delta']
        self.len_wind  = kwargs['len_wind'] 
        self.wgt_diff  = kwargs['wgt_diff']

        # state / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.res_pred_tmp = []

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
    def _detect_drift(p_idx, p_max, s_idx, s_max, wgt_diff, len_wind, delta):
        """
        Evaluate the state of the concept

        :param p_idx:       acc. of  ML model untill idx-th point           (type: float)
        :param p_max:       max. acc. of ML model observed so far           (type: float)
        :param s_idx:       std. dev. of the ML model untill idx-th point   (type: float)
        :param s_max:       std. dev. of the ML model in respect to p_max   (type: float)
        :param wgt_diff:    weight difference (for arithmetic scheme)       (type: float)
        :param len_wind:    length of the sliding window containing prediction results (rng: >= 1)  (type: int)
        :param delta:       conf. value for FPR bound (rng: 0 ~ 1)          (type: float)
        :return:            state of the concept
        """
        state = 0 # 0: stable, 2: drift

        # compute test statistics (Bhattacharyya distance)
        term_1    = math.log(1/4 * ((s_max/s_idx)**2 + (s_idx/s_max)**2 + 2))
        term_2    = (p_max - p_idx)**2 / (s_max**2 + s_idx**2)
        test_stat = 1/4 * (term_1 + term_2) - 1/len_wind

        # compute alpha for drift detection
        numer = -math.log10((1 + wgt_diff)/2) * len_wind
        denom = np.sum([wgt_diff * idx for idx in range(1, len_wind + 1)])
        alpha = numer/(denom + 1e-9)

        if (p_max > p_idx) and (p_max - p_idx >= p_max * delta):
            # evaluate state
            state = 2 if test_stat >= alpha else 0
        # end if

        return state
    
    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
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
            tr_start_idx = tr_start_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set drift index
            drift_idx = te_end_idx

            # set the start index of model update
            tr_start_idx = drift_idx - min_len_tr 

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.res_pred_tmp = []
        # end if

        return tr_start_idx
    
    def _reset_parameters(self): # utilized in detect_drift
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_max = 0
        self.s_max = 0

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
            # set test set / time index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)
            te_time_idx  = X.iloc[te_start_idx:te_end_idx].index

            # create training/test set
            X_tr, y_tr = X.iloc[tr_start_idx:tr_end_idx], y[tr_start_idx:tr_end_idx]
            X_te, y_te = X.iloc[te_start_idx:te_end_idx], y[te_start_idx:te_end_idx]

            # train ml model and predict testset
            ml_mdl, y_pred_te = run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl)

            # extract prediction results
            if prob_type == 'CLF':
                res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
            elif prob_type == 'REG':
                res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
            # end if

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(te_time_idx)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.len_wind:
                # compute the weight for each entry in sliding window
                entry_wgt_list = [1 + (i - 1) * self.wgt_diff for i in range(1, self.len_wind + 1)]
    
                # compute the weighted accuracy and standard deviation till idx-th point
                numer = np.sum([entry_wgt_list[idx]*self.res_pred_tmp[idx] for idx in range(self.len_wind)])
                denom = np.sum(entry_wgt_list)
                p_idx = numer/(denom + 1e-9)
                s_idx = np.sum([(entry_wgt_list[idx]*self.res_pred_tmp[idx]-p_idx)**2 for idx in range(self.len_wind)])/(denom + 1e-9)

                # update p_max and s_max if p_max <= p_idx
                self.p_max = p_idx if self.p_max <= p_idx else self.p_max
                self.s_max = s_idx if self.p_max <= p_idx else self.s_max

                # evaluate state
                self.state = self._detect_drift(p_idx    = p_idx, 
                                                p_max    = self.p_max, 
                                                s_idx    = s_idx,
                                                s_max    = self.s_max,
                                                wgt_diff = self.wgt_diff,
                                                len_wind = self.len_wind,
                                                delta    = self.delta)

                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 min_len_tr   = min_len_tr,
                                                 tr_start_idx = tr_start_idx, 
                                                 tr_end_idx   = tr_end_idx, 
                                                 te_end_idx   = te_end_idx)
                
                # drop entry into sliding window
                self.res_pred_tmp = self.res_pred_tmp[len_batch:]
            # end if

            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None

