##################################################################################################################################################################
# PROJECT: TM_MLOps
# CHAPTER: Algorithm
# SECTION: Concept Drift Adaptation
# AUTHOR: Yang et al.
# DATE: since 24.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import pandas as pd
import numpy as np
import math

from sklearn.metrics import accuracy_score

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
# concept drift adaptation methods
##################################################################################################################################################################

class RecentPeriod:
    def __init__(self, det_mdl, res_det: dict, init_tr_end_idx: int):
        """
        Intialize values
        :param det_mdl:         concept drift detection model
        :param res_det:         result of concept drift detection result
        :param init_tr_end_idx: initial end index of training set
        """
        self.adapt_prd_list   = det_mdl.adapt_prd_list
        self.state            = res_det['state']     ### 250113 DHY: state 추가
        self.start_idx        = res_det['start_idx']
        self.end_idx          = res_det['end_idx']
        self.init_tr_end_idx  = init_tr_end_idx
        self.min_num_tr       = det_mdl.min_num_tr
    
    def set_adaptation_period(self):
        """
        :return adpat_prd: adaptation period (adapt_start_idx ~ adapt_end_idx)
        """
        # set the reference point
        if self.adapt_prd_list == []: # first adaptation
            # initial end index of training set
            ref_idx = self.init_tr_end_idx
        else:
            # end index of the previous adaptation
            ref_idx = self.adapt_prd_list[-1][1] 
        # end if
        
        ### 250113 DHY: stable일때 재학습 구간을 NONE ~ NONE으로 설정
        # set the start/end index of adaptation
        if self.state == 'stable':
            adapt_start_idx, adapt_end_idx = 'NONE', 'NONE'
        else:
            adapt_start_idx = ref_idx + self.start_idx
            adapt_end_idx   = ref_idx + self.end_idx

            # adjust the starting point of adaptation
            # to ensure the min. number of training data points    
            num_adapt = adapt_end_idx - adapt_start_idx # ex) 50 = 150 - 100
            if num_adapt < self.min_num_tr: # 50 < 100
                # ex) adapt_start_idx: 100 -> 50 (= 100 - 100 + 50)
                adapt_start_idx = adapt_start_idx - self.min_num_tr + num_adapt  
            # end if
        # end if

        # append adaptation period
        self.adapt_prd_list.append([adapt_start_idx, adapt_end_idx])

        return adapt_start_idx, adapt_end_idx
    
class BufferPeriod:
    def __init__(self):
        pass

    @staticmethod
    def set_adaptation_period():

        return None

class ReferencePeriod:
    def __init__(self):
        pass

    @staticmethod
    def set_adaptation_period():

        return None