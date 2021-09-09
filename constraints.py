import torch.nn as nn
import torch.nn.functional as F
from objectives import Margin

class Constraint(nn.Module):
    '''
    Base class for constraints functions g(x) <= 0.
    '''
    pass


class ConstraintMisclassify(Constraint):
    '''
    Other class than x0 must be predicted.
    
    y = class(x0)
    prediction = softmax(model(x))
    max_{i not y}(prediction[i]) >= prediction[y]

    g(x) := -Margin(prediction, y) <= 0
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


class ConstraintClassifyTarget(Constraint):
    '''
    Target class y must be predicted.

    prediction = softmax(model(x))
    max_{i not y}(prediction[i]) >= prediction[y]

    g(x) := Margin(prediction, y) <= 0
    '''

    def __init__(self, y, model, softmax=True):
        super().__init__()
        self.margin = Margin()
        self.y = y
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
        return self.margin(prediction, y)


class ConstraintDistance(Constraint):
    '''
    Prescribed maximal distance via distance(x, x0) <= epsilon.
    
    g(x) := distance(x, x0) - epsilon <= 0
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