import torch
import torch.nn as nn
import torch.nn.functional as F

class Margin(nn.Module):
    '''
    Objective(x) = Margin(prediction, k)

    Margin(prediction, k) = max_{i not k}(prediction[i]) - prediction[k]
    '''
    def forward(self, prediction, k):
        assert prediction.shape[0] == k.shape[0]
        assert prediction.ndim <= 2

        mask = torch.full_like(prediction, False, dtype=torch.bool)
        mask[range(prediction.shape[0]), k] = True

        prediction_other = prediction[~mask].view(prediction.shape[0], -1)
        prediction_k = prediction[mask]
        return prediction_other.max(1)[0] - prediction_k


class CrossEntropyLossOH(nn.Module):
    '''
    Equivalent to Pytorch CrossEntropyLoss for one-hot target y.
    '''
    def forward(self, prediction, y):
        ce = torch.sum(-y*F.log_softmax(prediction, dim=1), dim=1)
        return torch.mean(ce)


class Objective(nn.Module):
    '''
    Objective(x) = loss(model(x), y) if targetted or 
    Objective(x) = -loss(model(x), y) if not targetted.
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
            return self.loss(prediction, self.y)
        else:
            return -self.loss(prediction, self.y)