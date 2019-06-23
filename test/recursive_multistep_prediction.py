# -*- coding: utf-8 -*-
# @Time         : 2019-06-17 20:25
# @Author       : YIN Jiaming
# @Email        : 14jiamingyin@tongji.edu.cn
# @Filename     : recursive_multistep_prediction.py
# @IDE          : PyCharm


from utils.metrics import *
from models.deepmodels import *
from utils.data_utils import *


# Parameters
dataset = 'commodity'
model = 'LSTNet'
filename = '../save/{}_{}_1.pt'.format(dataset, model)
data_path = '../dataset/' + dataset + '.npy'
window = 30
horizon = 12
batch_size = 128
seed = 12345
gpu = 1
cuda = 0
# Load data.
Data = Data_utility(data_path, 0.6, 0.2, 0, horizon, window, 2)

def recurrent_multistep(model=None, data=None):
    """
    Recurrent multi-step predict using models trained by torch.
    :param model: Trained one-step model.
    :param input: Test inputs.
    :param horizon: max predict step.
    :return:
    """
    model.eval()
    predict = None
    test = None
    for X, Y in data.get_batches(data.test[0], data.test_m, batch_size, False):
        pred = torch.zeros(([X.size(0), data.h, data.m]))
        pred = Variable(pred)
        scale0 = data.scale.expand(Y.size(0), data.h, data.m)
        for i in range(data.h):
            output = model(X)
            scale = data.scale.expand(output.size(0), data.m)
            X.data[:, :-1, :] = X.data[:, 1:, :]
            X.data[:, -1, :] = output.data
            pred.data[:, i, :] = (output * scale).data

        if predict is None:
            predict = pred
            test = Y * scale0
        else:
            predict = torch.cat((predict, pred))
            test = torch.cat((test, Y*scale0))

    preds = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    return preds, Ytest

# Load model.
with open(filename, 'rb') as f:
    model = torch.load(f)
preds, Ytest = recurrent_multistep(model, Data)

time_step = [0, 2, 5, 8, 11]
rse = np.zeros(len(time_step))
rmse = np.zeros(len(time_step))
mape = np.zeros(len(time_step))
mae = np.zeros(len(time_step))


for i, t in enumerate(time_step):
    v = Ytest[:, t, :].reshape(-1)
    v_ = preds[:, t, :].reshape(-1)
    rse[i] = RSE(v, v_)
    rmse[i] = RMSE(v, v_)
    mae[i] = MAE(v, v_)
    mape[i] = MAPE(v, v_)

# Print results
print(dataset)
print(model)
print(time_step)

print('RSE:\t', rse)
print('RMSE:\t', rmse)
print('MAPE:\t', mape)
print('MAE:\t', mae)