##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from src.common.config import *
from src.algorithm.ml.ML import *

import pandas as pd
import numpy as np
import math
import time
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
# No update
##################################################################################################################################################################

class NO_UPDATE:
    def __init__(self, X, y, prob_type, train_len, model_type):
        self.X = X
        self.y = y
        self.prob_type = prob_type # Regression : 'reg', Classification : 'clf'
        self.train_len = train_len
        self.model_type = model_type # Scikit-learn GridSearch : 'slg', AutoML : 'aut'

    def build_model(self, tr_start, num_tr):
        def _mean_absolute_percentage_error(y_test, y_pred):
            return np.mean(np.abs((np.array(y_test) - np.array(y_pred))/np.array(y_test))) # MAPE 함수

        mape_scorer = make_scorer(_mean_absolute_percentage_error, greater_is_better = False) # 사용자 정의 스코어러 생성
     
        tr_end   = tr_start + num_tr
        te_start = tr_end
        
        if (self.prob_type == 'reg'):
            models = {
                'Lasso': Lasso(max_iter=10000),   
                'Decision Tree Regressor': DecisionTreeRegressor()
            }
            param_grids = {
                'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]},
                'Decision Tree Regressor': {'max_depth': [None, 10, 20, 30, 40, 50]}
            }

        elif (self.prob_type == 'clf'):
            models = {
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Logistic Regression': LogisticRegression(max_iter=10000),
            }
            param_grids = {
                'Decision Tree Classifier': {'max_depth': [None, 10, 20, 30, 40, 50]},
                'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']},
            }
            
        else:
            raise ValueError("Invalid prob_type. Choose 'clf' or 'reg'.")
            
        results = {}
        
        # training set
        X_tr, y_tr = self.X[tr_start:tr_end], self.y[tr_start:tr_end] 

        # normalize dataset
        scale     = MinMaxScaler()
        X_tr_norm = scale.fit_transform(X_tr)
        
        # convert 2d into 1d
        y_tr = np.array(y_tr).ravel()

        results = {}
        
        best_score = np.inf if self.prob_type == 'reg' else -np.inf
        best_model_name = None
        best_model = None
        best_params_overall = None
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
    
            # 회귀와 분류에 따른 분리
            if model_name in ['Lasso']:
                x_train = X_tr_norm
                y_train = y_tr
                scoring = mape_scorer
            elif model_name in ['Decision Tree Regressor']:
                x_train = X_tr
                y_train = y_tr
                scoring = mape_scorer
            elif model_name in ['Logistic Regression']:
                x_train = X_tr_norm
                y_train = y_tr
                scoring = 'accuracy'
            else:
                x_train = X_tr
                y_train = y_tr
                scoring = 'accuracy'
            
            # 교차 검증을 통해 최적의 파라미터 찾기
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring=scoring)
            grid_search.fit(x_train, y_train)
            
            # 최적의 파라미터와 모델 저장
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_
            results[model_name] = {
                'best_params': best_params,
                'best_model': best_estimator
            }
            
            if scoring == 'accuracy':
                score = grid_search.best_score_
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = best_estimator
                    best_params_overall = best_params
            else:
                score = -grid_search.best_score_  # MAPE는 negative로 반환되므로 부호 변경
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = best_estimator
                    best_params_overall = best_params
                        
        print(f"\nBest model: {best_model_name}")
        print(f"Best hyperparameters: {best_params_overall}")
        print(f"Best cross-validation score: {best_score}")
        print('*' * 100)
            
        # build model
        if best_model_name in ['Lasso', 'Logistic Regression']:
            best_model.fit(X_tr_norm, y_tr)
        else:
            best_model.fit(X_tr, y_tr)

        if (self.prob_type == 'reg'):
            y_tr_pred = best_model.predict(X_tr_norm)
            in_sample_error = _mean_absolute_percentage_error(y_tr, y_tr_pred)
        else:
            y_tr_pred = best_model.predict(X_tr_norm)
            in_sample_error = accuracy_score(y_tr, y_tr_pred)

        print('train set - in_sample_error : ', in_sample_error)
        print('*' * 100)

        performance_list = []
        
        for i in range(1, len(self.X) - num_tr + 1): 
            # test set
            X_te, y_te = self.X[te_start:te_start + i], self.y[te_start:te_start + i] 
            
            # convert 2d into 1d
            y_te = np.array(y_te).ravel()

            # normalize dataset
            X_te_norm = scale.transform(X_te)
            
            # predict test set
            y_pred = best_model.predict(X_te_norm)
            
            if (self.prob_type == 'reg'):
                performance = _mean_absolute_percentage_error(y_pred, y_te)
            else:
                performance = accuracy_score(y_pred, y_te)
            
            performance_list.append(performance)
            
        cwd = os.getcwd()
        path = cwd + "/result/result_GJK/no_update_result.csv"
        result = pd.DataFrame({'y' : y_te, 'y_hat' : y_pred})
        result.to_csv(path, index = False)

        print('test set - performance : ', np.mean(performance_list))
        print('*' * 100)
        
        return performance_list, in_sample_error
    
    def plot_performance(self, performance_list, in_sample_error, title):
        
        x1 = list(range(1, self.train_len + 1))
        x2 = list(range(self.train_len + 1,len(self.X) + 1))

        in_sample_error_ls = [in_sample_error] * self.train_len
        
        fig, ax1 = plt.subplots()

        ax1.plot(x1, in_sample_error_ls, 'b--')
        ax1.plot(x2, performance_list, 'r')
        ax1.set_xlabel('data index')
        plt.legend(('trainset performance', 'testset performance'))
        plt.title(title)
        
    #     plt.show()
    
##################################################################################################################################################################
# Periodic update
##################################################################################################################################################################

class PERIODIC_UPDATE:
    def __init__(self, X, y, prob_type, init_train_len, period, option):
        self.X = X
        self.y = y
        self.prob_type = prob_type # Regression : 'reg', Classification : 'clf'
        self.init_train_len = init_train_len
        self.period = period
        self.option = option # Pruning : 'prn', Cumulative : 'cum'

    def get_best_testset_size(self, te_sizes=range(300, 501, 5)):
        
        def _mean_absolute_percentage_error(y_test, y_pred):
            return np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / np.array(y_test))) # MAPE 함수
        
        X = self.X
        y = self.y
        
        init_train_len = self.init_train_len
        
        best_score = float('inf') if self.prob_type == 'reg' else 0
        best_te_size = 0

        for num_te in te_sizes:
            print(num_te)
            start = time.time()

            tr_start = 0
            tr_end = tr_start + init_train_len
            te_start = tr_end
            te_end = tr_end + num_te

            y_pred_list = []
            y_te_list = []

            while te_end <= X.shape[0]:
                X_tr, y_tr = X[tr_start:tr_end], y[tr_start:tr_end]
                X_te, y_te = X[te_start:te_end], y[te_start:te_end]

                y_tr = np.array(y_tr).ravel()
                y_te = np.array(y_te).ravel()

                scale = MinMaxScaler()
                X_tr_norm = scale.fit_transform(X_tr)
                X_te_norm = scale.transform(X_te)

                if self.prob_type == 'clf':
                    model = LogisticRegression()
                else:
                    model = LinearRegression()

                model.fit(X_tr_norm, y_tr)
                y_pred = model.predict(X_te_norm)

                y_pred_list.append(y_pred)
                y_te_list.append(y_te)

                tr_start, tr_end = tr_start, te_end
                te_start, te_end = te_start + num_te, te_end + num_te

            y_pred = np.array(y_pred_list).reshape(-1)
            y_te = np.array(y_te_list).reshape(-1)

            if self.prob_type == 'clf':
                score = accuracy_score(y_te, y_pred)
                is_better = score > best_score
            else:
                score = _mean_absolute_percentage_error(y_te, y_pred)
                is_better = score < best_score

            if is_better:
                best_score = score
                best_te_size = num_te

            end = time.time()
            print(num_te)
            print('Task type:', self.prob_type)
            print('Test set size:', num_te)
            print('Score:', round(score, 2))
            print('Time taken:', f"{end - start:.5f} sec")
            print('*' * 100)
            print(y_te.shape, y_pred.shape)
            

        return best_te_size
        

    def build_model(self):
        def _mean_absolute_percentage_error(y_test, y_pred):
            return np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / np.array(y_test))) # MAPE 함수

        mape_scorer = make_scorer(_mean_absolute_percentage_error, greater_is_better = False) # 사용자 정의 스코어러 생성
        
        tr_start = BLIND_ADAPT['INIT_TR_START']
        
        num_tr   = self.period if self.period != None else PERIODIC_UPDATE.get_best_testset_size(self)
        print(num_tr)
        
        X = self.X
        y = self.y
        
        init_train_len = self.init_train_len
    
        tr_start = 0
        tr_end = tr_start + init_train_len
        te_start = tr_end
        te_end = tr_end + num_tr

        y_pred_list = []
        y_te_list = []

        performance_list = []
    
        while te_end <= self.X.shape[0]:
            X_tr, y_tr = X[tr_start:tr_end], y[tr_start:tr_end]
            X_te, y_te = X[te_start:te_end], y[te_start:te_end]

            y_tr = np.array(y_tr).ravel()
            y_te = np.array(y_te).ravel()

            scale = MinMaxScaler()
            X_tr_norm = scale.fit_transform(X_tr)
            X_te_norm = scale.transform(X_te)
        
            if (self.prob_type == 'reg'):
                models = {
                    'Lasso': Lasso(max_iter=10000),   
                    'Decision Tree Regressor': DecisionTreeRegressor()
                }
                param_grids = {
                    'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]},
                    'Decision Tree Regressor': {'max_depth': [None, 10, 20, 30, 40, 50]}
                }

            elif (self.prob_type == 'clf'):
                models = {
                    'Decision Tree Classifier': DecisionTreeClassifier(),
                    'Logistic Regression': LogisticRegression(max_iter=10000),
                }
                param_grids = {
                    'Decision Tree Classifier': {'max_depth': [None, 10, 20, 30, 40, 50]},
                    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']},
                }
                
            else:
                raise ValueError("Invalid prob_type. Choose 'clf' or 'reg'.")
            
            results = {}
        
            best_score = np.inf if self.prob_type == 'reg' else -np.inf
            best_model_name = None
            best_model = None
            best_params_overall = None
            if tr_end == init_train_len:
                in_sample_error = best_score
                
        
            for model_name, model in models.items():
                print(f"Training {model_name}...")
        
                # 회귀와 분류에 따른 분리
                if model_name in ['Lasso']:
                    x_train = X_tr_norm
                    y_train = y_tr
                    scoring = mape_scorer
                elif model_name in ['Decision Tree Regressor']:
                    x_train = X_tr
                    y_train = y_tr
                    scoring = mape_scorer
                elif model_name in ['Logistic Regression']:
                    x_train = X_tr_norm
                    y_train = y_tr
                    scoring = 'accuracy'
                else:
                    x_train = X_tr
                    y_train = y_tr
                    scoring = 'accuracy'
                
                # 교차 검증을 통해 최적의 파라미터 찾기
                grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring=scoring)
                grid_search.fit(x_train, y_train)
                
                # 최적의 파라미터와 모델 저장
                best_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_
                results[model_name] = {
                    'best_params': best_params,
                    'best_model': best_estimator
                }
                
                if scoring == 'accuracy':
                    score = grid_search.best_score_
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model = best_estimator
                        best_params_overall = best_params
                else:
                    score = -grid_search.best_score_  # MAPE는 negative로 반환되므로 부호 변경
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model = best_estimator
                        best_params_overall = best_params
                            
            print(f"\nBest model: {best_model_name}")
            print(f"Best hyperparameters: {best_params_overall}")
            print(f"Best cross-validation score: {best_score}")
            print('*' * 100)
                
            # build model
            if best_model_name in ['Lasso', 'Logistic Regression']:
                best_model.fit(X_tr_norm, y_tr)
            else:
                best_model.fit(X_tr, y_tr)

            if (self.prob_type == 'reg'):
                y_tr_pred = best_model.predict(X_tr_norm)
                in_sample_error = _mean_absolute_percentage_error(y_tr, y_tr_pred)
            else:
                y_tr_pred = best_model.predict(X_tr_norm)
                in_sample_error = accuracy_score(y_tr, y_tr_pred)

            print('train set - in_sample_error : ', in_sample_error)
            print('*' * 100)

            # convert 2d into 1d
            y_te = np.array(y_te).ravel()
            print(y_te.shape)
            # normalize dataset
            X_te_norm = scale.transform(X_te)
            
            # predict test set
            y_pred = best_model.predict(X_te_norm)
            print(y_pred.shape)
            
            y_pred_list.extend(list(y_pred))
            y_te_list.extend(list(y_te))
            
            print(tr_start, tr_end, te_start, te_end, len(y_pred_list), len(y_te_list))
            
            tr_start, tr_end = tr_start if self.option == 'cum' else tr_start + num_tr, te_end
            te_start, te_end = te_start + num_tr, te_end + num_tr
        
        return y_pred_list, y_te_list, in_sample_error
    
    def plot_performance(self, y_pred_list, y_te_list, in_sample_error, title):
        
        x1 = list(range(1, self.init_train_len + 1))
        score = []
        for i in range(len(y_te_list)):
            score.append(accuracy_score(y_pred_list[:i], y_te_list[:i]))
            
        x2 = list(range(self.init_train_len + 1, self.init_train_len + len(y_te_list) + 1))

        in_sample_error_ls = [in_sample_error] * self.init_train_len
        
        fig, ax1 = plt.subplots()

        ax1.plot(x1, in_sample_error_ls, 'b--')
        ax1.plot(x2, score, 'r')
        ax1.set_xlabel('data index')
        plt.legend(('trainset performance', 'testset performance'))
        plt.title(title)
        
        plt.show()

