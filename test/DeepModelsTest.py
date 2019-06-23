import argparse
import time
from utils.data_utils import *
from utils.optim import *
from models.deepmodels import *
from utils.metrics import *


params={
    'data': '../dataset/commodity.npy',
    'horizon': 1,
    'window': 30,
    'highway_window': 14,
    'skip': -1,
    'model': 'LSTNet',
    'CNN_kernel': 2,
    'hidRNN': 50,
    'hidCNN': 50,
    'hidSkip': 0,
    'L1Loss': False,
    'epochs': 50,
    'batch_size': 64,
    'output_fun': 'linear',
    'dropout': 0.2,
    'save': '../save/commodity_LSTNet_1.pt',
    'clip': 10,
    'seed': 12345,
    'log_interval': 2000,
    'optim': 'adam',
    'lr': 0.001,
    'normalize': 2,
    'gpu':0,
    'cuda':0
}


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, ifsave=False, ds='ds'):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data[0]
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data[0]
        n_samples += (output.size(0) * data.m)

        if predict is None:
            predict = output*scale
            test = Y * scale
        else:
            predict = torch.cat((predict, output*scale))
            test = torch.cat((test, Y*scale))

    rse = math.sqrt(total_loss / n_samples) / data.rse
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    v = Ytest.reshape(-1)
    v_ = predict.reshape(-1)
    mae = MAE(v, v_)
    mape = MAPE(v, v_)
    rmse = RMSE(v, v_)

    return rse, rmse, mape, mae


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data[0]
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples

Data = Data_utility(params['data'], 0.6, 0.2, params['cuda'], params['horizon'], params['window'], params['normalize'])
print(Data.rse)

model = eval(params['model'])(params, Data)

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if params['L1Loss']:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)
if params['cuda']:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

best_val = 10000000
optim = Optim(
    model.parameters(), params['optim'], params['lr'], params['clip'],
)

# At any point you can hit Ctrl + C to break out of training early.
all_start = time.time()
try:
    print('begin training')
    for epoch in range(1, params['epochs'] + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, params['batch_size'])
        val_rse, val_rmse, val_mape, val_mae = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               params['batch_size'])
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rsme {:5.4f} | valid mape  {:5.4f} | valid mae  {:5.4f} '.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rmse, val_mape, val_mae))
        # Save the model if the validation loss is the best we've seen so far.

        if val_rse < best_val:
            with open(params['save'], 'wb') as f:
                torch.save(model, f)
            best_val = val_rse
        if epoch % 5 == 0:
            test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     params['batch_size'])
            print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae {:5.4f}".format(test_rse, test_rmse, test_mape, test_mae))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
all_end = time.time()

# Load the best saved model.
with open(params['save'], 'rb') as f:
    model = torch.load(f)
test_rse, test_rmse, test_mape, test_mae = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                    params['batch_size'])
print("test rse {:5.4f} | test rmse {:5.4f} | test mape {:5.4f} | test mae  {:5.4f}".format(test_rse, test_rmse, test_mape, test_mae))

