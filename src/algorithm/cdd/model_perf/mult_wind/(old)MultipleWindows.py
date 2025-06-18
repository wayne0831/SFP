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
import scipy.stats as stats
from scipy.stats import cramervonmises_2samp, ks_2samp, anderson_ksamp

from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
#from src.util.util import run_ml_model_pipeline
from datetime import datetime, timedelta

import matplotlib.pyplot as pipet
import matplotlib.patches as patches
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
# Multiple Windows-based Methods
##################################################################################################################################################################

##################################################################################################################################################################
# ADWIN (ADaptive WINdowing, 2007) 
##################################################################################################################################################################

class ADWIN:
    def __init__(self, **kwargs):
        """
        Initialize values
         **kwargs: {delta: float, wind_mem: int, clt_idx: int, mgn_rate: float}

        :param delta:       conf. value for FPR bound                      (rng: 0 ~ 1) (type: float)
        :param wind_mem:    window memory parameter controlling step size  (rng: >= 1)  (type: int)
        :param clt_idx:     min. num. of data points to obey CLT           (rng: >= 1)  (type: int)
        :param mgn_rate:    ratio of cut indices used in cut_idx_list      (rng: 0 ~ 1) (type: float)
        """
        # hyperparameters of ADWIN
        self.delta      = kwargs['delta']     
        self.wind_mem   = kwargs['wind_mem']
        self.clt_idx    = kwargs['clt_idx']
        self.mgn_rate   = kwargs['mgn_rate']

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
    def _generate_cut_indices(res_pred_te_idx, wind_mem, mgn_rate):
        """
        Generate cut points to divide the window into left-/right- sub window
        Note that med_idx, mgn_rate is not in ADWIN. -> 'mgn_rate = 1' is ADWIN setting
        These two variables are added to generate more realistic cut points.

        :param res_pred_te_idx: window of prediction results                                (type: list)
        :param wind_mem:        window memory parameter controlling step size  (rng: >= 1)  (type: int)
        :param mgn_rate:        ratio of cut indices used in cut_idx_list      (rng: 0 ~ 1) (type: float)
        :return cut_idx_list:   set of cut points of given window
        """
        cut_idx_list = []

        # compute initial cut_idx (c)
        c, sqr = 1 + 1/wind_mem, 0
        cut_idx = np.floor(np.power(c, sqr)) # floor(c^square) == 1 
        end_idx = len(res_pred_te_idx)
        while cut_idx < end_idx:
            cut_idx_list.append(int(cut_idx))
            sqr    += 1
            cut_idx = np.floor(np.power(c, sqr))  # floor(c^square)    
        # end while 

        # remove dupipeicated elements and order them in descending way
        cut_idx_list = sorted(set(cut_idx_list), reverse=True)

        ### DHY: mgn 도입
        ### 기존 ADWIN은 모든 cut_idx를 다 search함, 하지만 이는 오랜 시간 소요를 초래함
        ### 따라서 window의 중간 지점을 기준으로 mgn을 도입해 search하는 지점의 하/상한선을 제한함
        # set the median index and margin of cut_idx_list
        med_idx = np.searchsorted(cut_idx_list, np.median(cut_idx_list))
        mgn     = int((end_idx * mgn_rate) // 2)

        # set the start/end index of cut_idx_list
        start_idx, end_idx = (med_idx - mgn), (med_idx + mgn)

        # extract cut indices which satisfies the range condition
        cut_idx_list = cut_idx_list[start_idx:end_idx]

        return cut_idx_list

    @staticmethod
    def _detect_drift(res_pred_te_idx, cut_idx, delta):
        """
        Evaluate the state of the concept

        :param res_pred_te_idx: window of prediction results            (type: list)     
        :param cut_idx:         cut point                               (type: inst)
        :param delta:           conf. value for FPR bound  (rng: 0 ~ 1) (type: float)
        :return:                state of the concept
        """
        state = 0 # 0: stable, 2: drift

        # set left/right sub-window
        left_wind  = res_pred_te_idx[:cut_idx]
        right_wind = res_pred_te_idx[cut_idx:]

        # compute test statistics for drift detection
        mean_left_wind   = np.mean(left_wind)        # mean (= acc) of left sub-window
        mean_right_wind  = np.mean(right_wind)       # mean (= acc) of right sub-window
        test_stat        = np.abs(mean_left_wind - mean_right_wind) 
    
        # compute epsilon cut (=eps_cut) for the drift detection threshold
        len_left_wind   = len(left_wind) 
        len_right_wind  = len(right_wind) 
        len_wind        = len_left_wind + len_right_wind 
        har_mean        = 1/(1/len_left_wind + 1/len_right_wind) # harmomnic mean

        # compute eps_cut
        var      = np.var(res_pred_te_idx)   # variance
        delta    = delta/np.log(len_wind)    # modified delta
        eps_cut  = np.sqrt(2/har_mean * var * np.log(2/delta)) + (2/(3*har_mean) * np.log(2/delta))
        
        # evaluate state of the concept
        state = 2 if test_stat >= eps_cut else 0
        
        return state

    def _adapt_drift(self, state, min_len_tr, cut_idx, tr_start_idx, tr_end_idx, te_end_idx):
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
            # set drift index
            drift_idx = cut_idx

            # set the start index of model update
            if (te_end_idx - drift_idx) <= min_len_tr:
                tr_start_idx = te_end_idx - min_len_tr 
            else: 
                tr_start_idx = te_end_idx - drift_idx
            # end if

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(te_end_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.res_pred_tmp = []

            print(f'Drift detected at {tr_start_idx}')
        # end if

        return tr_start_idx

    def _reset_parameters(self): # utilized in detect_drift
        """
        Reset parameters for next iteration of concept drift detection
        No parameters to be reset in ADWIN
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

            if len(self.res_pred_tmp) >= self.clt_idx:
                # generate cut indices
                cut_idx_list = self._generate_cut_indices(res_pred_te_idx = self.res_pred_tmp,
                                                          wind_mem        = self.wind_mem,
                                                          mgn_rate        = self.mgn_rate)

                for cut_idx in cut_idx_list:
                    # evaluate state of the concept
                    self.state = self._detect_drift(res_pred_te_idx = self.res_pred_tmp, 
                                                    cut_idx         = cut_idx, 
                                                    delta           = self.delta)

                    # set the start index of updated training set
                    tr_start_idx = self._adapt_drift(state        = self.state, 
                                                     min_len_tr   = min_len_tr,
                                                     cut_idx      = cut_idx,
                                                     tr_start_idx = tr_start_idx, 
                                                     tr_end_idx   = tr_end_idx, 
                                                     te_end_idx   = te_end_idx)
                    if self.state == 2:
                        self._reset_parameters()    
                        break # end detection process                       
                # end for
            # end if

            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None

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
# DOER (Dynamic and On-line Ensemble Regression, 2014) 
##################################################################################################################################################################
"""
class DOER:
    def __init__(self, **kwargs):
        # hyperparameters of DOER
        self.alpha         = kwargs['alpha'] # 0.04 on real-world datasets
        self.len_wind      = kwargs['len_wind'] 
        self.max_num_ensmb = kwargs['max_num_ensmb'] # 15

        # parameters used in drift adaptation
        #self.adapt_prd_list  = [] # set of adaptation periods
        #self.min_num_tr      = kwargs['min_num_tr']

        # windows of predicted/real target values in test set
        self.y_pred_te = np.array([]) 
        self.y_real_te = np.array([])  

        # set of learners (ensembles) and weight
        self.ensmb     = []
        self.wgt_list  = []
        self.life_list = []
        self.mse_list  = []
        self.err_list  = []

    #  def run_cdda(self, X, y, scaler, prob_type, ml_mdl, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd):
    def run_doer(self, X, y, scaler, prob_type, ml_mdl, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd):
        
        # 회귀 문제일 경우, 예측 오차 판단 임계치
        num_data = len(X)

        # 초기 학습: 초기 윈도우로 모델 학습
        X_tr, y_tr = X.iloc[tr_start_idx:tr_end_idx], y[tr_start_idx:tr_end_idx]
        ml_mdl.fit(X_tr, y_tr)

        #
        self.ensmb.append(ml_mdl)
        self.wgt_list.append(1)
        self.life_list.append(0)
        self.mse_list.append(0)

        while tr_end_idx < num_data:
            # set test set index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)

            # create test set
            X_te, y_te = X.iloc[te_start_idx:te_end_idx], y[te_start_idx:te_end_idx]

            print('1111111111111111')
            print(len(self.ensmb))
            print(np.array([ml_mdl.predict(X_te).tolist() for ml_mdl in self.ensmb])))
            

            # predict test set
            ### todo 250310 DHY: 개념적으로는 여기에 np.mean이 들어가면 안됨
            #y_te_preds    = np.array([ml_mdl.predict(X_te).reshape(-1)[0] for ml_mdl in self.ensmb])
            y_te_preds    = np.array([np.mean([ml_mdl.predict(X_te).tolist() for ml_mdl in self.ensmb])])
            wgt_y_te_pred = np.dot(np.array(self.wgt_list), y_te_preds) / sum(self.wgt_list)

            print(y_te_preds, wgt_y_te_pred)

            # print('*' * 100)
            # print(f'idx/data_idx {idx}/{det_end_idx}')
            # print(f'Prediction: {wgt_y_te_pred} / Weight sum: {sum(self.wgt_list)}')
            # print(f'num ml models: {len(self.ensmb)}')

            # compute error for each ml model
            err_list_idx = [(y_te - y_te_pred) ** 2 for y_te_pred in y_te_preds]

            self.err_list.append(err_list_idx)
            if len(self.err_list) > self.len_wind:
                self.err_list.pop(0)

            # compute MSE for each ml model 
            for i in range(len(self.ensmb)):
                if self.life_list[i] == 0:
                    self.mse_list[i] = 0
                elif self.life_list[i] <= self.len_wind:
                    self.mse_list[i] = (self.life_list[i] - 1)/self.life_list[i] * self.mse_list[i] + err_list_idx[i]/self.life_list[i]
                else:
                    self.mse_list[i] = self.mse_list[i] + (err_list_idx[i] - self.err_list[0][i])/self.len_wind
                # end if
                
                self.life_list[i] += 1
            # end for

            # update weight for each ml model
            mse_med       = np.median(self.mse_list)
            self.wgt_list = list(np.exp(-(np.array(self.mse_list) - mse_med) / mse_med + 1e-9))

            #idx = idx + 1

            # update ml models
            #if abs((wgt_y_te_pred - y_te)/y_te) > self.alpha:
            if abs((wgt_y_te_pred - np.mean(y_te))/np.mean(y_te)) > self.alpha:
#                tr_start_idx = init_tr_start_idx + idx
#                tr_end_idx   = init_tr_end_idx + idx

                tr_start_idx = tr_start_idx + len_batch
                tr_end_idx   = tr_end_idx + len_batch

                X_tr = X.iloc[tr_start_idx:tr_end_idx]
                y_tr = y[tr_start_idx:tr_end_idx]

                # build and train ml model
                ml_mdl.fit(X_tr, y_tr) 

                self.ensmb.append(ml_mdl)
                self.wgt_list.append(1)
                self.life_list.append(0)
                self.mse_list.append(0)

                if len(self.ensmb) > self.max_num_ensmb:
                    worst_idx = np.argmax(self.mse_list)  # MSE가 가장 높은 모델 찾기

                    print(f'worst_idx: {worst_idx}')

                    del self.ensmb[worst_idx]
                    del self.wgt_list[worst_idx]
                    del self.life_list[worst_idx]
                    del self.mse_list[worst_idx]
                # end if
            # end if

            # append predictions/targets to predefined windows
            self.y_pred_te = np.append(self.y_pred_te, wgt_y_te_pred)
            self.y_real_te = np.append(self.y_real_te, y_te)

            
        # end while
        
        return None
"""

##################################################################################################################################################################
# WSTD (Wilcoxon Rank Sum Test Drift Detector, 2017) 
##################################################################################################################################################################

class WSTD:
    def __init__(self, **kwargs):
        """
        Initialize values
         **kwargs: {alpha_w: float, alpha_d: float, len_sw: int, max_len_lw: int}

        :param alpha_w:     warning sig.level   (rng: 0 ~ 1)    (type: float)
        :param alpha_d:     drift sig.level     (rng: 0 ~ 1)    (type: float)
        :param len_sw:      length of short window for calculating the recent accuracy (rng: >= 1)          (type: int)
        :param max_len_lw:  maximum length of long window for calculating the overall accuracy (rng: >= 1)  (type: int)
        """
        # hyperparameters of WSTD
        self.alpha_w    = kwargs['alpha_w']
        self.alpha_d    = kwargs['alpha_d']
        self.len_sw     = kwargs['len_sw']
        self.max_len_lw = kwargs['max_len_lw'] 

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
    def _detect_drift(res_pred_lw, res_pred_sw, alpha_w, alpha_d):
        """
        Evaluate the state of the concept

        :param res_pred_lw:   window of prediction results for long window    till idx-th point (type: list)        
        :param res_pred_sw:   window of prediction results for short window   till idx-th point (type: list)         
        :param alpha_w:     warning sig.level   (rng: 0 ~ 1)    (type: float)
        :param alpha_d:     drift sig.level     (rng: 0 ~ 1)    (type: float)
        :return:            state of the concept
        """
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        # compute # of corrections and errors among the window
        n_o  = len(res_pred_lw)      # length of long window 
        r_o  = np.sum(res_pred_lw)   # num. of corrections among the long window
        w_o  = n_o - r_o             # num. of errors among the long window

        n_r  = len(res_pred_sw)      # length of short window
        r_r  = np.sum(res_pred_sw)   # num. of corrections among the short window
        w_r  = n_r - r_r             # num. of errors among the long window
        
        # compute test statistic
        r_rank = (1 + r_o + r_r) / 2
        w_rank = r_o + r_r + ((1 + w_o + w_r) / 2)
        sum_o  = r_rank * r_o + w_rank * w_o
        sum_r  = r_rank * r_r + w_rank * w_r

        sum_rank  = sum_o if sum_o < sum_r else sum_r
        aux       = n_o + n_r + 1
        test_stat = (sum_rank - n_r*aux/2) / np.sqrt(n_o * n_r * aux/12)

        # compute two-tailed p-value
        p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))

        # evaluate state
        state = 0 if p_val >= alpha_w else 1 if p_val >= alpha_d else 2

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
        No parameters to be reset in WSTD
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

            if len(self.res_pred_tmp) >= 2*self.len_sw:
                # set windows of older/recent prediction results
                res_pred_lw = self.res_pred_tmp[(-self.len_sw-self.max_len_lw):-self.len_sw]
                res_pred_sw = self.res_pred_tmp[-self.len_sw:]

                # evaluate state of the concept
                self.state = self._detect_drift(res_pred_lw  = res_pred_lw, # prediction result for lw 
                                                res_pred_sw  = res_pred_sw, # prediction result for sw
                                                alpha_w      = self.alpha_w,
                                                alpha_d      = self.alpha_d)

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
# FDD (Fisher Drift Detector, 2018) 
##################################################################################################################################################################

class FDD:
    def __init__(self, **kwargs):
        """
        Initialize values
         **kwargs: {alpha_w: float, alpha_d: float, len_sw: int, type: str, min_num_tr: int}

        :param alpha_w:     warning sig.level   (rng: 0 ~ 1)    (type: float)
        :param alpha_d:     drift sig.level     (rng: 0 ~ 1)    (type: float)
        :param len_sw:      length of short window for calculating the recent accuracy (rng: >= 1) (type: int)
        :param type:        method to compute p-value (p: FPDD, s: FSDD, t: FTDD)   (type: str)
        """
        # hyperparameters of FDD
        self.alpha_w  = kwargs['alpha_w']
        self.alpha_d  = kwargs['alpha_d']
        self.len_sw   = kwargs['len_sw'] 
        self.type     = kwargs['type']

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
    def _detect_drift(res_pred_lw, res_pred_sw, alpha_w, alpha_d, type, fact_list, const_f):
        """
        Evaluate the state of the concept

        :param res_pred_lw:   window of prediction results for long window    till idx-th point (type: list)        
        :param res_pred_sw:   window of prediction results for short window   till idx-th point (type: list)         
        :param alpha_w:     warning sig.level   (rng: 0 ~ 1)    (type: float)
        :param alpha_d:     drift sig.level     (rng: 0 ~ 1)    (type: float)
        :param type:        method to compute p-value (p: FPDD, s: FSDD, t: FTDD)   (type: str)
        :param fact_list:   list containing the factorial values for each integer (0 to 2*len_sw)   (type: list)
        :param const_f:     contant for fisher test (type: float)
        :return:            state of the concept
        """
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        # compute # of corrections and errors among the window
        n_o  = len(res_pred_lw)    # length of long window
        r_o  = np.sum(res_pred_lw) # num. of corrections among the long window
        w_o  = n_o - r_o           # num. of errors among the long window

        w   = len(res_pred_sw)     # length of short window
        r_r = np.sum(res_pred_sw)  # num. of corrections among the short window
        w_r = w - r_r              # num. of errors among the short window

        # compute adjusted corrections (w_p) & errors (r_p) among the long window
        w_p = int(w_o * w/n_o)
        r_p = w - w_p         

        # compute test statistic and p-value
        if (w_r < 5 or r_r < 5 or w_p < 5 or r_p < 5) or (type == 't'): # FTDD
            left_opnd  = fact_list[w_r + w_p] / fact_list[w_r] / fact_list[w_p]
            right_opnd = fact_list[r_r + r_p] / fact_list[r_r] / fact_list[r_p]
            test_stat  = left_opnd * right_opnd
            p_val      = test_stat * const_f * 2
        elif type == 'p': # FPDD
            numer     = np.abs(w_p - w_r) - 1
            denom     = np.sqrt((w_p + w_r) * (2*w - (w_p + w_r))/2*w) 
            test_stat = numer/denom
            p_val     = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        elif type == 's': # FSDD
            ew_r = (w_r + w_p) * (w_r + r_r) / 2*w
            er_r = (r_r + r_p) * (w_r + r_r) / 2*w
            ew_p = (w_r + w_p) * (w_p + r_p) / 2*w
            er_p = (r_r + r_p) * (w_p + r_p) / 2*w

            left_opnd  = pow(np.abs(w_r - ew_r), 2)/ew_r +  pow(np.abs(r_r - er_r), 2)/er_r
            right_opnd = pow(np.abs(w_p - ew_p), 2)/ew_p +  pow(np.abs(r_p - er_p), 2)/er_p
            test_stat  = left_opnd + right_opnd
            p_val      = 1 - stats.chi2.cdf(test_stat, 1)
        # end if

        # evaluate state
        state = 0 if p_val >= alpha_w else 1 if p_val >= alpha_d else 2

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
        No parameters to be reset in FDD
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
        # compute factorial list and constant for fisher test
        fact_list = [math.factorial(i) for i in range(2*self.len_sw + 1)]
        const_f   = pow(fact_list[self.len_sw], 2)/fact_list[2*self.len_sw]

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

            if len(self.res_pred_tmp) >= 2*self.len_sw:
                # set windows of older/recent prediction results
                res_pred_lw = self.res_pred_tmp[:-self.len_sw]
                res_pred_sw = self.res_pred_tmp[-self.len_sw:]

                # evaluate state of the concept
                self.state = self._detect_drift(res_pred_lw  = res_pred_lw, # prediction result for lw 
                                           res_pred_sw  = res_pred_sw, # prediction result for sw
                                           alpha_w      = self.alpha_w,
                                           alpha_d      = self.alpha_d,
                                           type         = self.type,
                                           fact_list    = fact_list,
                                           const_f      = const_f)

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
# MR-DDM (Multi-Resolution Drift Detection Methpd, 2025) 
##################################################################################################################################################################

### TODO
class MRDDM:
    def __init__(self, **kwargs):
        """
        Intialize values
         **kwargs: {alpha_w: float, alpha_d: float, len_sw: int}

        :param alpha_w:     warning sig.level   (rng: 0 ~ 1)
        :param alpha_d:     drift sig.level     (rng: 0 ~ 1)
        :param len_sw:      length of short window for calculating the recent accuracy (rng: >= 1) 
        """
        # hyperparameters of STEPD
        self.len_step = kwargs['len_step'] 
        self.alpha_d  = kwargs['alpha_d'] 
        self.len_sw   = kwargs['len_sw'] 

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
    def _detect_drift(res_pred_tr, res_pred_lw, res_pred_sw, alpha_d):
        """
        Evaluate the state of the concept

        :param res_pred_lw:   window of prediction results for long window till idx-th point    (type: list)
        :param res_pred_sw:   window of prediction results for short window till idx-th point   (type: list)
        :param alpha_w:       warning sig.level   (rng: 0 ~ 1)    (type: float)
        :param alpha_d:       drift sig.level     (rng: 0 ~ 1)    (type: float)
        :return state:        state of the concept
        """
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        # p_val_lw = cramervonmises_2samp(res_pred_tr, res_pred_lw).pvalue
        # p_val_sw = cramervonmises_2samp(res_pred_tr, res_pred_sw).pvalue

        # 0319: input 30, len_sw: 36, len_step: 3 => ctq: 91.83
        _, p_val_lw = ks_2samp(res_pred_tr, res_pred_lw)
        _, p_val_sw = ks_2samp(res_pred_tr, res_pred_sw)

        # p_val_lw = anderson_ksamp([res_pred_tr, res_pred_lw]).significance_level
        # p_val_sw = anderson_ksamp([res_pred_tr, res_pred_sw]).significance_level

        # evaluate state
        if (p_val_lw < alpha_d) and (p_val_sw >= alpha_d):
            state = 1 # gradual drift
        elif (p_val_lw >= alpha_d) and (p_val_sw < alpha_d):
            state = 2 # abrupt drift
        else:
            state = 0
        # end if

        return state

    def _adapt_drift(self, state, res_pred_lw, res_pred_sw, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
        """
        Adjust the training set for ml model update
        
        :param state:           state of the concept                 (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param tr_start_idx:    previous start index of training set (type: int)
        :param tr_end_idx:      previous end index of training set   (type: int)        
        :param te_end_idx:      previous end index of test set       (type: int)                
        :return tr_start_idx:   updated start index of training set  (type: int)
        """
        if state == 0:  # stable
            tr_start_idx = tr_start_idx
        else:
            print(f'Drift detected at {te_end_idx} / state: {state}')

            # set drift index
            drift_idx = te_end_idx

            if state == 1: # gradual drift
                # set the start index of model update
                tr_start_idx = drift_idx - len(res_pred_lw*self.len_step) 
            elif state == 2: # abrupt drift
                # set the start index of model update
                tr_start_idx = drift_idx - len(res_pred_sw)                 
            # end if

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
            # set test set / time index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, num_data)
            te_time_idx  = X.iloc[te_start_idx:te_end_idx].index

            # create training/test set
            X_tr, y_tr = X.iloc[tr_start_idx:tr_end_idx], y[tr_start_idx:tr_end_idx]
            X_te, y_te = X.iloc[te_start_idx:te_end_idx], y[te_start_idx:te_end_idx]

            # train ml model and predict testset
            ml_mdl, y_pred_te = run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl)

            # compute lists of training loss
            y_pred_tr   = ml_mdl.predict(X_tr)
            tr_err_list = [abs(pred - real) for pred, real in zip(y_pred_tr, y_tr)]

            # # extract prediction results
            # if prob_type == 'CLF':
            #     res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
            # elif prob_type == 'REG':
            #     res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
            # # end if

            res_pred_idx = [abs(pred - real) for pred, real in zip(y_pred_te, y_te)] 

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(te_time_idx)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.len_sw*self.len_step:
                # set windows of older/recent prediction results

                # sampling
                res_pred_lw = self.res_pred_tmp[::self.len_step] 



                # aggregation
                #res_pred_lw = [np.mean(self.res_pred_tmp[i:i+self.len_step]) for i in range(0, len(self.res_pred_tmp), self.len_step)]

                res_pred_sw = self.res_pred_tmp[-self.len_sw:]

                print(f'res_pred_lw: {len(res_pred_lw)} / res_pred_sw: {len(res_pred_sw)}')

                # evaluate state of the concept
                self.state = self._detect_drift(res_pred_tr = tr_err_list,
                                                res_pred_lw = res_pred_lw, # prediction result for lw 
                                                res_pred_sw = res_pred_sw, # prediction result for sw
                                                alpha_d     = self.alpha_d)

                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 res_pred_lw  = res_pred_lw,
                                                 res_pred_sw  = res_pred_sw,
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