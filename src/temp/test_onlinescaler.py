##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from src.util.preprocess import *
from river.datasets import synth
import numpy as np

from src.algorithm.cdd.model_perf.sing_wind import DDM

# 1. 데이터 스트림 로딩 (Agrawal with concept drift)
#dataset = synth.Agrawal(classification_function=1, seed=42).take(10000)
dataset = synth.LED(noise_percentage=0.2, seed=42).take(10000)

X, y = convert_synthethic_dataset_to_array(data=dataset)


prob_type = 'CLF'
perf_bnd  = 1


# 초기화
scaler  = OnlineStandardScaler()
ml_mdl  = GaussianNB()

param   = {'alpha_w': 2, 'alpha_d': 3, 'warm_start': 30}
cdd_mdl = DDM()

y_real_list = []
y_pred_list = []

# 온라인 학습 루프
batch_size = 1  # 또는 mini-batch 크기
num_data   = len(X)
for idx in range(0, num_data, batch_size):
    print(f'start_idx: {idx} / end_idx: {min(idx + batch_size, num_data)}')
    start_idx, end_idx = idx, min(idx + batch_size, num_data)
    X_idx = X[start_idx:end_idx]
    y_idx = y[start_idx:end_idx]

    # partially scale dataset
    scaler.partial_fit(X_idx)
    X_scl = scaler.transform(X_idx)

    # partially fit the ml model
    ml_mdl.partial_fit(X_scl, y_idx, classes=np.unique(y)) if start_idx == 0 else ml_mdl.partial_fit(X_scl, y_idx)

    # predict the dataset
    y_pred_idx = ml_mdl.predict(X_scl)

    if prob_type == 'CLF':
        res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_idx, y_idx)] 
    elif prob_type == 'REG':
        res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_idx, y_idx)] 
    # end if

    y_pred_list.extend([int(x) for x in y_pred_idx])
    y_real_list.extend(y_idx)

    if cdd_mdl.state == 2: # drift (DDM은 state = 1인 경우도있고... 모든 기법을 포괄해야하는데...)
        scaler = OnlineStandardScaler()
        ml_mdl = GaussianNB()


    
    #    


    # print(y_pred)
# end for


print(222222222222222222)
# print(len(y), len(y_pred_list))
print(len(y_real_list))
print(accuracy_score(y_real_list, y_pred_list))
# print(33333333333333333333)
# print(y_pred)

