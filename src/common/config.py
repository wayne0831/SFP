##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Common
# SECTION: Configuration
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from sklearn.linear_model   import LinearRegression, Lasso, LogisticRegression, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes    import GaussianNB
#from skmultiflow.trees      import HoeffdingTreeRegressor

from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.algorithm.cdd.model_perf.sing_wind.SingleWindow import *
from src.algorithm.cdd.model_perf.mult_wind.MultipleWindows import *
from src.algorithm.cda.CDA  import *
import os
from river.datasets import synth

##################################################################################################################################################################
# version control
##################################################################################################################################################################

DATE            = 'SFP'
DATA_TYPE       = 'APP'   
DATA            = 'PDX'   
PROB_TYPE       = 'REG'   
ML_METH         = 'SGD'   

#  'DDM', 'STEPD', 'FHDDMS', 'MRDDM'
CDD_METH_LIST   = ['MRDDM']

CDA_METH_LIST   = ['REC']
VER             = 'V4'    # PDX -> v1: n_m_m[:39] / v2: n_m_m[:19] / v3: posco

SCALER          = OnlineStandardScaler() if ML_METH == 'SGD' else StandardScaler()
LEN_BATCH       = 12
MIN_LEN_TR      = 1

##################################################################################################################################################################
# paths
##################################################################################################################################################################

# data path
DATA_PATH = {
    'APP': { # application
        'PDX':   'data/application/df_sfp.csv'
    }
}

# result path
RES_PATH = {
    'PERF_ROOT': f'result/performance/',
    'PRED_ROOT': f'result/prediction/',
    'CDDA':      f'cdda/',
}

##################################################################################################################################################################
# datasets
##################################################################################################################################################################

DATASET = {
    'APP': { # application
        'PDX' : {
            'INDEX': 'A', # 1 data point: 10 minutes
            'INPUT_V4' : [ # v4: n_m_m[:30]
                'C', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'P', 'Z', 'AB',
                'CU', 'CW', 'CX', 'CY', 'CZ', 'DA', 'DB', 'DC', 'DF', 'DG', 
                'DH', 'DI', 'DJ', 'DM', 'DN', 'DO', 'DQ', 'DR', 'DW'
            ],
            'TRGT': ['target'],
            'PROB_TYPE': 'REG' # REG, CLF
        }
    }
}

##################################################################################################################################################################
# algorithms
##################################################################################################################################################################

# machine learning
ML = {
    'REG': {
        'SGD':   SGDRegressor(random_state=42, max_iter=1000, tol=1e-3),
    }
}

# concept drift detection
CDD = {
    # single window
    'DDM':    DDM,       # type: ignore
    'STEPD':  STEPD,     # type: ignore
    'FHDDMS': FHDDMS,     # type: ignore
    'MRDDM':  MRDDM,     # type: ignore
}

# concept drift adaptation
CDA = {
    'REC':  RecentPeriod,   # type: ignore
    'BUF':  BufferPeriod,   # type: ignore
    'REF':  ReferencePeriod # type: ignore
}

##################################################################################################################################################################
# experiments - grid search
##################################################################################################################################################################

# concept drift detection
CDD_PARAM_GRID = {
    # single window
    'DDM': {
        'alpha_w':      [2],  # [1.5, 2],            
        'alpha_d':      [3],  # [2.5, 3],
        'clt_idx':      [30], # [30, 60],
    },
    'STEPD': {
        'alpha_w':      [0.05],            
        'alpha_d':      [0.003], # [0.005, 0.1],
        'len_sw':       [30], # [30, 60],
    },
    'FHDDMS': {
        'delta':        [1e-7], # [1e-7, 1e-6, 1e-5],
        'len_lw':       [100],  # [100, 200],
        'len_sw':       [30],   # [30, 60], PDX: 25
    },
    'MRDDM': {
        'alpha_d':      [0.05],
        'len_step':     [3],
        'len_sw':       [30],
    }
}

##################################################################################################################################################################
# experiment
##################################################################################################################################################################

# informed (non-periodic) adaptation
INFRM_ADAPT = {
    'APP':{
        'PDX': {
            'INIT_TR_START_IDX': 0, # 37241: 2023-01-01 0:10
            'INIT_NUM_TR':       37241, # 37241: 2023-01-01 0:10
        }
    }
}