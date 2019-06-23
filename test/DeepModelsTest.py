"""
Deep Models:
    * NN (Naive)
    * MLP
    * CNN
    * RNN
    * RNN (LSTM)
    * RNN (GRU)
    * LSTNet (ref: https://github.com/laiguokun/LSTNet)
Implemented by PyTorch.
"""

import torch
from torch import nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, args, data):
        super(NN, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear = nn.Linear(self.P * self.m, self.m)
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        c = x.view(-1, self.P*self.m)
        return self.output(self.linear(c))

class MLP(nn.Module):
    def __init__(self, args, data):
        super(MLP, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)

        self.n1 = int(self.P * self.m * 1.5)
        self.n2 = self.n1

        self.dense1 = nn.Linear(self.P * self.m, self.n1)
        self.dense2 = nn.Linear(self.n1, self.n2)
        self.dense3 = nn.Linear(self.n2, self.m)

        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        x = x.view(-1, self.P * self.m)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.output(self.dense3(x))
        if self.output:
            x = self.output(x)
        return self.dropout(x)


class RNN(nn.Module):
    def __init__(self, args, data):
        super(RNN, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
        self.RNN = nn.RNN(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidR, self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        _, h = self.RNN(x)
        h = h.contiguous().view(-1, self.hidR)
        h = self.dropout(h)
        o = self.linear(h)
        if self.output:
            o = self.output(o)
        return o

class GRU(nn.Module):
    def __init__(self, args, data):
        super(GRU, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
        self.RNN = nn.GRU(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidR, self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        _, h = self.RNN(x)
        h = h.view(-1, self.hidR)
        h = self.dropout(h)
        o = self.linear(h)
        if self.output:
            o = self.output(o)
        return o


class LSTM(nn.Module):
    def __init__(self, args, data):
        super(LSTM, self).__init__()
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidR = args.hidRNN
        self.lstm = nn.LSTM(
            input_size=self.m,
            hidden_size=self.hidR,
            batch_first=True
        )
        self.linear = nn.Linear(self.P * self.hidR, self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        # x     [batch_size, time_step, input_size]
        # r_out [batch_size, time_step, output_size]
        # h_n   [n_layers, batch_size, hidRNN]
        # h_c   [n_layers, batch_size, hidRNN]
        r_out, (h_n, h_c) = self.lstm(x)
        r_out = r_out.contiguous().view(-1, self.P * self.hidR)
        r_out = self.dropout(r_out)
        o = self.linear(r_out)
        if self.output:
            o = self.output(o)
        return o


class CNN(nn.Module):
    def __init__(self, args, data):
        super(CNN, self).__init__()
        self.P = args.window
        self.m = data.m
        self.dropout = nn.Dropout(p=args.dropout)
        self.hidC = args.hidCNN
        self.Ck = args.CNN_kernel
        self.width = self.m
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.width))
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear = nn.Linear(self.hidC * (self.P - self.Ck + 1) * (self.m - self.width + 1), self.m)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh


    def forward(self, x):
        # x: [batch_size, window, N(=m)]
        # c: [batch_size, window, 1, N]
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        # [batch_size, hid_CNN, window-height+1, N-width+1] CNN_kernel:(height, width)
        c = self.dropout(c)
        c = c.view(-1, self.hidC * (self.P - self.Ck + 1)*(self.m-self.width+1))
        # output: [batch_size, N]

        c = self.linear(c)
        if self.output:
            c = self.output(c)
        return c


"""
    Ref: https://github.com/laiguokun/LSTNet
         https://arxiv.org/abs/1703.07015
    Implemented by PyTorch.
"""

class LSTNet(nn.Module):
    def __init__(self, args, data):
        super(LSTNet, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if self.output:
            res = self.output(res)
        return res

