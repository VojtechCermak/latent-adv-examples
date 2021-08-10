import torch
import torch.nn as nn
import torch.nn.functional as F
from objectives import Margin

class Constraint(nn.Module):
    '''
    Base class for constraints functions g(x).
    '''
    pass


class ConstraintMisclassify(Constraint):
    '''
    Wraps margin function as constraint function g(x).
    g(x) < 0 if model classifies x differently than x0.
    '''
    def __init__(self, x0, model, softmax=True):
        super().__init__()
        self.margin = Margin()
        self.x0 = x0
        self.y = model(x0).argmax(1)
        self.model = model
        self.softmax = softmax

    def forward(self, x, subset=None):
        if subset is not None:
            y = self.y[subset]
        else:
            y = self.y

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and subset of x0 have different batch size.')

        prediction = self.model(x)
        if self.softmax:
            prediction = F.softmax(prediction, 1)
        return -self.margin(prediction, y)


class ConstraintDistance(Constraint):
    '''
    Wraps distance function d(x0, x) as constraint function g(x).
    g(x) < 0 if dist(x, x0) < epsilon.
    '''
    def __init__(self, x0, distance, epsilon):
        super().__init__()
        self.x0 = x0
        self.distance = distance
        self.epsilon = epsilon

    def forward(self, x, subset=None):
        if subset is not None:
            x0 = self.x0[subset]
        else:
            x0 = self.x0

        if x.shape[0] != x0.shape[0]:
            raise ValueError('x and subset of x0 have different batch size.')

        return self.distance(x, x0) - self.epsilon