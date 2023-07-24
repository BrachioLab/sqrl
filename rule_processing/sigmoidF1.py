import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class sigmoidF1(nn.Module):

    def __init__(self, S = -100, E = -0.5):
        super(sigmoidF1, self).__init__()
        self.S = S
        self.E = E

    @torch.cuda.amp.autocast()
    def forward(self, y_hat, y):
        
        # y_hat = torch.sigmoid(y_hat)

        b = torch.tensor(self.S)
        c = torch.tensor(self.E)

        sig = 1 / (1 + torch.exp(b * (y_hat + c)))

        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)

        sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        # cost = 1 - sigmoid_f1
        macroCost = torch.mean(sigmoid_f1)

        return macroCost

class macroSoftF1(nn.Module):

    def __init__(self):
        super(macroSoftF1, self).__init__()

    @torch.cuda.amp.autocast()
    def forward(self, y_hat, y):
        
        y_hat = torch.sigmoid(y_hat)

        tp = torch.sum(y_hat * y, dim=0)
        fp = torch.sum(y_hat * (1 - y), dim=0)
        fn = torch.sum((1 - y_hat) * y, dim=0)

        macroSoft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - macroSoft_f1
        macroCost = torch.mean(cost)

        return macroCost