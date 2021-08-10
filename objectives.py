import torch
import torch.nn as nn
import torch.nn.functional as F

class Objective(nn.Module):
    '''
    Wraps Pytorch loss function in form of f(x, y) as objective in form of f(x).
    '''
    def __init__(self, y, loss, model, targeted):
        super().__init__()
        self.y = y
        self.loss = loss
        self.model = model
        self.targeted = targeted

    def forward(self, x):
        prediction  = self.model(x)
        if self.targeted:
            return self.loss(prediction , self.y)
        else:
            return -self.loss(prediction , self.y)


class Margin(nn.Module):
    '''
    Calculates margin: max_{i not y}(x[i]) - x[y]
    '''
    def forward(self, prediction, y):
        assert prediction.shape[0] == y.shape[0]
        assert prediction.ndim <= 2

        mask = torch.full_like(prediction, False, dtype=torch.bool)
        mask[range(prediction.shape[0]), y] = True

        prediction_other = prediction[~mask].view(prediction.shape[0], -1)
        prediction_y = prediction[mask]
        return prediction_other.max(1)[0] - prediction_y


class CrossEntropyLossOH(nn.Module):
    '''
    Equivalent to Pytorch CrossEntropyLoss for one-hot target y.
    '''
    def forward(self, prediction, y):
        ce = torch.sum(-y*F.log_softmax(prediction, dim=1), dim=1)
        return torch.mean(ce)