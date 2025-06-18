##################################################################################################################################################################
# PROJECT: TM_MLOps
# CHAPTER: Concept Drift Detection and Adaptation
# SECTION: Algorithm
# SUB-SECTION: Machine Learning Algorithms
# AUTHOR: Yang et al.
# DATE: since 24.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

##################################################################################################################################################################
# Machine Learning Algorithms
##################################################################################################################################################################

class MachineLearning:
    def __init__(self):
        pass
    
class PerformanceMetric:
    def __init__(self):
        pass
    
    # MAPE
    def mean_absolute_percentage_error(y_test, y_pred):
        return np.mean(np.abs((np.array(y_test) - np.array(y_pred))/np.array(y_test)))

    # CTQ
    def hit_ratio(y_true, y_pred, std):
        hit = abs(np.array(y_true) - np.array(y_pred)) < std
        return hit.sum()/len(hit)*100