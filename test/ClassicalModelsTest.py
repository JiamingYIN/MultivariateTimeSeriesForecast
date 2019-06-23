# -*- coding: utf-8 -*-
# @Time         : 2019-06-09 15:25
# @Author       : YIN Jiaming
# @Email        : 14jiamingyin@tongji.edu.cn
# @Filename     : regmodels_test.py
# @IDE          : PyCharm


from utils.load_data import *
from models.ClassicalModels import *

# Dataset: electricity.txt, exchange_rate.txt, solar_AL.txt, traffic.txt, PeMS.csv, commodity.npy(pkl), congestion.npy(pkl)
dataset = '../dataset/ts.npy'

# data: T*N (T: length of time series, N: number of variables)
data = load_data(dataset, type='npy')

# Models
modelset = ['LR', 'SVR', 'Ridge', 'LASSO', 'SVR-polyK', 'RF', 'KNN']
modeltype = 'Ridge'

if modeltype not in modelset:
    raise NameError("Don't support this model!")

params = {'alpha': .2,              # Lasso, Ridge
          'n': 5,                   #
          'weight': 'distance',     # KNN: 'uniform', 'distance'
          'dist': 'canberra',       # KNN: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mhalanobis'
          'k_start': 12,            # KNN
          'k_end': 15,              # KNN
          'model': modeltype,
          'kernel': 'linear'}

history = 24
horizon = 24
T, N = data.shape

time_step = [0, 2, 5, 8, 11]

fcasts = np.zeros((0, horizon))
gts = np.zeros((0, horizon))

for n in range(N):
    if n % 10 == 0:
        print(n, 89*'=')
    temp = data[:, n]
    X, Y = gen_data_1d(temp, history=history, horizon=horizon)

    ds = {}
    n_samples = X.shape[0]
    tr = int(n_samples * 0.6)
    va = int(n_samples * 0.2)
    te = n_samples - tr - va

    scaler = None
    ds['train'] = [X[:tr, :], Y[:tr, 0:1]]
    ds['valid'] = [X[tr:tr + va, :], Y[tr + va, 0:1]]
    ds['test'] = [X[-te:, :], Y[-te:, :]]

    if modeltype == 'KNN':
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)
        X = scaler.transform(X)
        X[np.isnan(X)] = 0
        model = KNN_reg(data=ds, param=params)

    else:
        model = regression(data=ds, params=params)

    fcast = multi_step_predict(model=model, input=ds['test'][0], horizon=horizon, scaler=scaler)
    fcasts = np.concatenate((fcasts, fcast), axis=0)
    gts = np.concatenate((gts, ds['test'][1]), axis=0)


fcasts = fcasts[:, time_step]
gts = gts[:, time_step]
rse = np.zeros((len(time_step)))
rmse = np.zeros((len(time_step)))
mape = np.zeros((len(time_step)))
mae = np.zeros((len(time_step)))

for i in range(len(time_step)):
    rse[i] = RSE(gts[:, i], fcasts[:, i])
    rmse[i] = RMSE(gts[:, i], fcasts[:, i])
    mape[i] = MAPE(gts[:, i], fcasts[:, i])
    mae[i] = MAE(gts[:, i], fcasts[:, i])


print('Dataset:\t', dataset)
print("Model:\t", modeltype)
print("timestep:\t", time_step)
print("rse:\t", rse)
print("rmse:\t", rmse)
print("mape:\t", mape)
print("mae:\t", mae)


