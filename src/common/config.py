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
DATA_TYPE       = 'APP'   # 'APP', 'SYN', 'REAL'
DATA            = 'PDX'   # 'POSCO', 'PDX', 'LED'
PROB_TYPE       = 'REG'   # 'CLF', 'REG'
ML_METH         = 'SGD'   # LASSO, LOG_REG
CDD_METH_LIST   = ['MRDDM']
CDA_METH_LIST   = ['REC']
VER             = 'V4'    # PDX -> v1: n_m_m[:39] / v2: n_m_m[:19] / v3: posco

SCALER          = OnlineStandardScaler() if ML_METH == 'SGD' else StandardScaler()
LEN_BATCH       = 12
MIN_LEN_TR      = 1

SYN_DATA        = synth.LED(noise_percentage=0.1, seed=42).take(10000)

"""
250615
Completed: {
    SingleWindow:    ['DDM']
    MultipleWindows: ['STEPD', 'MRDDM']
        TBD: PL, AL
}

Ongoing: ['DOER']
"""
# CDD_METH_LIST   = ['DDM', 'FHDDM', 'MDDM', 'BDDM', 'ADWIN', 'STEPD', 'WSTD', 'FDD', 'FHDDMS']


##################################################################################################################################################################
# paths
##################################################################################################################################################################

# data path
DATA_PATH = {
    'SYN': { # synthetic
        'HYPER':  'data/synthetic/hyperplane/rotatingHyperplane.csv',
        'LED':    'data/synthetic/led/ledDriftSMall.csv',
        'MIXED':  'data/synthetic/mixedDrift/mixedDrift.csv',
        'MOVING': 'data/synthetic/movingSquares/movingSquares.csv',
        'M_RBF':  'data/synthetic/rbf/movingRBF.csv',
        'I_RBF':  'data/synthetic/rbf/interchangingRBF.csv',
        'SEA':    'data/synthetic/sea/SEA.csv',
    },
    'REAL': { # real-world
        'ELEC2':    'data/real_world/Elec2/df.csv',
        'AIR_POLUT':'data/real_world/airpollution/encoded_air_pollution.csv',
    },
    'APP': { # application
        'PDX':   'data/application/df_posco_ver3.csv',
        'POSCO': "data/application/posco_battery.csv"
    }
}

# pickle path
PKL_PATH = {}

# result path
RES_PATH = {
    'PERF_ROOT': f'result/performance/',
    'PRED_ROOT': f'result/prediction/',
    'ML':        f'ml/',
    'CDDA':      f'cdda/',
    # 'CDDA_DHY':  f'cdda/DHY/',
    # 'CDDA_GJK':  f'cdda/GJK/'
}

##################################################################################################################################################################
# datasets
##################################################################################################################################################################

DATASET = {
    'SYN': { # synthetic
        'HYPER': {
            'INDEX': 'INDEX',
            'INPUT': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'            
        },
        'LED': {
            'INDEX': None,
            'INPUT': [
                'att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 
                'att9', 'att10', 'att11', 'att12', 'att13', 'att14', 'att15', 'att16', 
                'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24'
                ],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'            
        },
        'MIXED': {
            'INDEX': 'INDEX',
            'INPUT': ['A', 'B'],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'            
        },
        'MOVING': {
            'INDEX': 'INDEX',
            'INPUT': ['A', 'B'],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'            
        },
        'M_RBF': {
            'INDEX': 'INDEX',
            'INPUT': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'            
        },
        'I_RBF': {
            'INDEX': 'INDEX',
            'INPUT': ['A', 'B'],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'
        },
        'SEA': {
            'INDEX': 'INDEX',
            'INPUT': ['A', 'B', 'C'],
            'TRGT':  ['class'],
            'PROB_TYPE': 'CLF'
        }
    },
    'APP': { # application
        'PDX' : {
            'INDEX': 'A', # 1 data point: 10 minutes
            'INPUT_V1': [ # v1: n_m_m[:39]
                'CX', 'G', 'E', 'H', 'AA', 'AG', 'O', 'I', 'AQ', 'N', 'C', 'AI', 'D', 
                'W', 'Q', 'F', 'DR', 'T', 'AN', 'DJ', 'R', 'DM', 'AO', 'DW', 'S', 'AC', 
                'U', 'J', 'AB', 'DK', 'Z', 'DE', 'DD', 'AJ', 'AE', 'AD', 'DQ', 'AK', 'Y'
                ],
            'INPUT_V2': [ # v2: n_m_m[:19]
                'CX', 'G', 'E', 'H', 'AA', 'AG', 'O', 'I', 'AQ', 
                'N', 'C', 'AI', 'D', 'W', 'Q', 'F', 'DR', 'T', 'AN'
                ], 
            'INPUT_V3': [ # v3: posco
                'C', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'P', 'Z', 'AB', 
                'CU', 'CW', 'CX', 'CY', 'CZ', 'DA', 'DB', 'DC', 'DF', 'DG', 
                'DH', 'DI', 'DJ', 'DM', 'DN', 'DO', 'DQ', 'DR', 'DW'
                ],
            'INPUT_V4' : [ # v4: n_m_m[:30]
                'C', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'P', 'Z', 'AB',
                'CU', 'CW', 'CX', 'CY', 'CZ', 'DA', 'DB', 'DC', 'DF', 'DG', 
                'DH', 'DI', 'DJ', 'DM', 'DN', 'DO', 'DQ', 'DR', 'DW'
            ],
            'TRGT': ['target'],
            'PROB_TYPE': 'REG' # REG, CLF
        },
        'POSCO': { # 1 data point: 1 minute
            'INDEX': 'TIME',
            'INPUT_V1': [
                "mR_CT_44201_PV", "mR_CT_44202_PV", "mR_CT_44203_PV", "mR_CT_44205_PV", "mR_CT_44206_PV",
                "mR_CT_44207_PV", "mR_CT_44209_PV", "mR_CT_44210_PV", "mR_CT_44211_PV", "mR_FT_44201_PV",
                "mR_FT_44202_PV", "mR_FT_44203_PV", "mR_FT_44204_PV", "mR_FT_44209_PV", "mR_FT_44210_PV",
                "mR_FT_44211_PV", "mR_FT_44214_PV", "mR_FT_44215_PV", "mR_FT_44216_PV", "mR_FT_44217_PV",
                "mR_FT_44218_PV", "DV_deltaP_AS_1", "DV_deltaP_AS_2", "DV_deltaP_AS_3", "DV_deltaP_BA_1",
                "DV_deltaP_BA_2", "DV_deltaP_BA_3", "DV_deltaP_SB_1", "DV_deltaP_SB_2", "DV_deltaP_SB_3"
            ],
            'INPUT_V2': [
                "mR_CT_44201_PV", "mR_CT_44202_PV", "mR_CT_44203_PV", "mR_CT_44205_PV", "mR_CT_44206_PV",
                "mR_CT_44207_PV", "mR_CT_44209_PV", "mR_CT_44210_PV", "mR_CT_44211_PV", "mR_FT_44201_PV",
                "mR_FT_44202_PV", "mR_FT_44203_PV", "mR_FT_44209_PV", "mR_FT_44210_PV", "mR_FT_44211_PV",
                "mR_FT_44214_PV", "mR_FT_44215_PV", "mR_FT_44216_PV", "mR_FT_44204_PV", "mR_FT_44217_PV",
                "mR_FT_44218_PV", "DV_deltaP_AS_1", "DV_deltaP_AS_2", "DV_deltaP_AS_3", "DV_deltaP_BA_1",
                "DV_deltaP_BA_2", "DV_deltaP_BA_3", "DV_deltaP_SB_1", "DV_deltaP_SB_2", "DV_deltaP_SB_3"
            ],
            'INPUT_V3': [
                "mR_PH_43102_PV", "mR_CT_43101_PV", "440T01_Li", "mR_CT_44201_PV", 
                "mR_CT_44202_PV", "mR_CT_44203_PV", "mR_CT_44205_PV", "mR_CT_44206_PV",
                "mR_CT_44207_PV", "mR_CT_44209_PV", "mR_CT_44210_PV", "mR_CT_44211_PV",
                "mR_FT_44201_PV", "mR_FT_44202_PV", "mR_FT_44203_PV", "mR_FT_44209_PV",
                "mR_FT_44210_PV", "mR_FT_44211_PV", "mR_FT_44214_PV", "mR_FT_44215_PV",
                "mR_FT_44216_PV", "mR_FT_44204_PV", "mR_FT_44217_PV", "mR_FT_44218_PV"
            ],
            'INPUT_V4': [
                "mR_PH_43102_PV", "mR_CT_43101_PV", "440T01_Li", "mR_CT_44201_PV", "mR_CT_44202_PV", 
                "mR_CT_44203_PV", "mR_CT_44205_PV", "mR_CT_44206_PV", "mR_CT_44207_PV", "mR_CT_44209_PV", 
                "mR_CT_44210_PV", "mR_CT_44211_PV", "mR_FT_44201_PV", "mR_FT_44202_PV", "mR_FT_44203_PV", 
                "mR_FT_44204_PV", "mR_FT_44209_PV", "mR_FT_44210_PV", "mR_FT_44211_PV", "mR_FT_44214_PV", 
                "mR_FT_44215_PV", "mR_FT_44216_PV", "mR_FT_44217_PV", "mR_FT_44218_PV"
            ],
            'TRGT_V1': ['442T05_Li'],
            'TRGT_V2': ['442T05_S'],
            'TRGT_V3': ['442T06_Li'],
            'TRGT_V4': ['442T06_S'],
            'PROB_TYPE': 'REG' # REG, CLF
        }
    },
    'REAL': { # real-world
        'ELEC2': { # 1 data point: 30 minutes
            'INDEX':     None,
            'INPUT_V1':  ['nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer'],
            'TRGT':      ['class'],
            'PROB_TYPE': 'CLF'
        },
        'AIR_POLUT': {
            'INDEX':     None,
            'INPUT':     ['dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain',
                            'wnd_dir_NE', 'wnd_dir_NW', 'wnd_dir_SE', 'wnd_dir_cv'],
            'TRGT':      ['pollution'],
            'PROB_TYPE': 'REG'            
        }
    }
}

##################################################################################################################################################################
# algorithms
##################################################################################################################################################################

# machine learning
ML = {
    'CLF': {
        'LOG_REG': LogisticRegression(),
        'NB':      GaussianNB(),
        'RF':      RandomForestClassifier(random_state=42, n_estimators=50),
        'XGB':     XGBClassifier(random_state=42, n_estimators=50),
        'MLP':     MLPClassifier(random_state=42)
    },
    'REG': {
        'LIN':   LinearRegression(),
        'SGD':   SGDRegressor(random_state=42, max_iter=1000, tol=1e-3),
        'LASSO': Lasso(alpha=0.5, random_state=42),
        'RIDGE': Ridge(),
        'RF':    RandomForestRegressor(random_state=42, n_estimators=50),
        'XGB':   XGBRegressor(random_state=42, n_estimators=50),
        'MLP':   MLPRegressor(random_state=42)
    }
}

# concept drift detection
CDD = {
    # single window
    'DDM':    DDM,       # type: ignore
    # #'EDDM':   EDDM,      # type: ignore
    # 'FHDDM':  FHDDM,     # type: ignore
    # #'RDDM':   RDDM,      # type: ignore   
    # 'MDDM':   MDDM,      # type: ignore
    # 'BDDM':   BDDM,      # type: ignore
    # #'VRDDM':  VRDDM,     # type: ignore
    # 'DDM_ATTN': DDM_ATTN, # type: ignore

    # double window
    # 'ADWIN':  ADWIN,     # type: ignore
    'STEPD':  STEPD,     # type: ignore
    # 'WSTD':   WSTD,     # type: ignore
    # 'FDD':    FDD,       # type: ignore
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

# machine learning
ML_PARAM_GRID = {  # 'ML': GJK

}

# concept drift detection
CDD_PARAM_GRID = {
    # single window
    'DDM': {
        'alpha_w':      [2],  # [1.5, 2],            
        'alpha_d':      [3],  # [2.5, 3],
        'clt_idx':      [30], # [30, 60],
    },
    # 'EDDM': {
    #     'alpha_w':      [0.95],            
    #     'alpha_d':      [0.90],  
    #     'clt_idx':      [30], # [30, 60],
    # },
    'FHDDM': {
        'delta':        [1e-7], # [1e-7]
        'len_wind':     [36],   # [30, 60], PDX: 25
    },
    # 'RDDM': {
    #     'alpha_w':      [2],  # [1.5, 2],            
    #     'alpha_d':      [3],  # [2.5, 3],
    #     'clt_idx':      [30], # [30, 60],
    #     'max_len':      [1000],
    #     'min_len':      [500],
    #     'warn_limit':   [50],
    # },
    'MDDM': {
        'delta':        [1e-7], # [1e-7, 1e-6, 1e-5],
        'len_wind':     [36], # [30, 60], PDX: 25
        'wgt_type':     ['a', 'g', 'e'], # ['a', 'g', 'e'],
        'wgt_diff':     [0.01], # [0.01, 0.02, 0.03],
        'wgt_rate':     [1.01], # [1.01, 1.5, 2, 2.5],
        'wgt_lambda':   [0.01], # [0.01, 0, 0.02],
    },
    'BDDM': {
        'delta':        [1e-7], # [0.25, 0.50]
        'len_wind':     [36], # [30, 60],
        'wgt_diff':     [0.01], # [0.01, 0.02, 0.03],
    },
    # double window
    'ADWIN': {
        'delta':        [0.1], # [0.1, 0.2, 0.3],
        'wind_mem':     [5],
        'mgn_rate':     [1], # [0.6, 0.8, 1],
        'clt_idx':      [36], # [30, 60],
    },
    'STEPD': {
        'alpha_w':      [0.05],            
        'alpha_d':      [0.003], # [0.005, 0.1],
        'len_sw':       [30], # [30, 60],
    },
    'WSTD': {
        'alpha_w':      [0.05],            
        'alpha_d':      [0.003], # [0.005, 0.1],
        'len_sw':       [36],    # [30, 60],
        'max_len_lw':   [500],   # [500],
    },
    'FDD': {
        'alpha_w':      [0.05],            
        'alpha_d':      [0.003], # [0.005, 0.1],
        'len_sw':       [36],    # [30, 60],
        'type':         ['t', 'p', 's'],   # ['t', 'p', 's']
    },
    'FHDDMS': {
        'delta':        [1e-7], # [1e-7, 1e-6, 1e-5],
        'len_lw':       [100],  # [100, 200],
        'len_sw':       [30],   # [30, 60], PDX: 25
    },

    # ### todo
    # 'VRDDM': {
    #     'len_wind':     [36],
    #     'conc':         [0.01], # 0.01 ~ 0.2
    #     'alpha_d':      [0.05]
    # },

    'MRDDM': {
        'alpha_d':      [0.05],
        'len_step':     [3],
        'len_sw':       [30],
    },
    # 'DDM_ATTN': {
    #     'alpha_w':      [2],  # [1.5, 2],            
    #     'alpha_d':      [3],  # [2.5, 3],
    #     'clt_idx':      [36], # [30, 60],
    #     #'conc':         [0.01], # 0.01 ~ 0.2
    #     'top_n':        [5]
    # },
}

##################################################################################################################################################################
# pipeline
##################################################################################################################################################################

PIPELINE_EXECUTION = {
    'RUN_EXP': False
}


##################################################################################################################################################################
# experiment
##################################################################################################################################################################

# constant (no adaptation)
CONST = {
    'TR_START'    : 0,
    'NUM_TR'      : 13500,
    'NUM_TR_RATE' : 0.3
}

# blind (periodic) adaptation
BLIND_ADAPT = {
    'INIT_TR_START'    : 0,
    'INIT_NUM_TR'      : 13500,
    'INIT_NUM_TR_RATE' : 0.3,
    'INIT_NUM_TE'      : 1500    # update period
}

# informed (non-periodic) adaptation
INFRM_ADAPT = {
    'APP':{
        'PDX': {
            'INIT_TR_START_IDX': 0, # 37241: 2023-01-01 0:10
            'INIT_NUM_TR':       37241, # 37241: 2023-01-01 0:10
        },
        'POSCO': {
            'INIT_TR_START_IDX': 0, 
            'INIT_NUM_TR':       16303, # Tr: 16303 / Te: 16304
        }
    },
    'REAL':{
        'ELEC2': {
            'INIT_TR_START_IDX': 0, 
            'INIT_NUM_TR':       13500,
            #'INIT_NUM_TR_RATE' : 0.3
        }
    }
}