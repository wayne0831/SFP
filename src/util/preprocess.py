##################################################################################################################################################################
# PROJECT: DIISERTATION
# CHAPTER: Util
# SECTION: preprocess
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
import time

##################################################################################################################################################################
# user-defined utility functions
##################################################################################################################################################################

class OnlineStandardScaler:
    def __init__(self):
        # set the initial statistics: the number of samples, running mean/variance
        self.num_data = 0
        self.mean     = None
        self.var      = None

    def partial_fit(self, X):
        """
        Incrementally update the mean and variance with new batch X.
        """
        X = np.asarray(X)
        
        if self.mean is None:
            # initialize mean and variance from the initial batch
            self.num_data = X.shape[0]
            self.mean     = X.mean(axis=0)
            self.var      = X.var(axis=0)
        else:
            # update statistics using batch statistics
            cum_num_data = self.num_data + X.shape[0]
            new_mean     = (self.num_data * self.mean + X.shape[0] * X.mean(axis=0)) / cum_num_data
            new_var      = (self.num_data * self.var  + X.shape[0] * X.var(axis=0))  / cum_num_data
            
            # update statistics
            self.num_data, self.mean, self.var = cum_num_data, new_mean, new_var
        # end if
        
        return self

    def transform(self, X):
        """
        Apply standard scaling (zero mean, unit variance) using running statistics.
        """
        X     = np.asarray(X)
        X_scl = (X - self.mean) / (np.sqrt(self.var) + 1e-8)  # epsilon to avoid division by zero

        return X_scl

    def fit_transform(self, X):
        """
        Convenience method to update statistics and return the scaled batch.
        """
        X_scl = self.partial_fit(X).transform(X)

        return X_scl
    
def set_synthethic_dataset(data):
    """
    Convert synthetic dataset into array type
    """
    # set X_list and y_list
    X_list, y_list = [], []
    for x, y in data:
        X_list.append(list(x.values()))
        y_list.append(y)

    # convert list into numpy array
    X = pd.DataFrame(X_list)
    y = np.array(y_list)

    return X, y