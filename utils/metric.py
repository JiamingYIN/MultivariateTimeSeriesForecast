import numpy as np


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


def RMSE(v, v_):
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    return np.mean(np.abs(v_ - v))


def MAPE(v, v_):
    return np.mean(np.abs(v_ - v) / v)


def get_acc():
    pass


def RSE(v, v_):
    return np.sqrt(np.mean((v_ - v)**2)) / normal_std(v)
