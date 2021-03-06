import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvergenceError(Exception):
    pass


class ProjectionIdentity(nn.Module):
    '''
    Project x to itself.
    '''
    def forward(self, x0, x):
        return x


class ProjectionL2(nn.Module):
    '''
    Project x to the Euclidean epsilon ball around x0.
    '''
    def __init__(self, eps):
        self.eps = eps

    def projection(self, x, eps):
        x_norm = x.norm(p=2, dim=tuple(range(1, x.ndim)), keepdim=True)
        x = torch.where(x_norm > eps, eps*(x / x_norm), x)
        return x

    def forward(self, x0, x):
        delta = x - x0
        delta_projected = self.projection(delta, self.eps)
        return x0 + delta_projected


class ProjectionLinf(nn.Module):
    '''
    Project x to the l-inf epsilon ball around x0.
    '''
    def __init__(self, eps):
        self.eps = eps

    def projection(self, x, eps):
        return x.clamp(-eps, eps)

    def forward(self, x0, x):
        delta = x - x0
        delta_projected = self.projection(delta, self.eps)
        return x0 + delta_projected


class ProjectionBinarySearch(nn.Module):
    '''
    Assumes constraint(x0) < 0.

    If constraint(x) < 0: Returns x.
    If constraint(x) >= 0: Returns x' between x and x0 with constraint(x') = 0.
    '''
    def __init__(self, constraint, threshold=0.001, max_steps=100):
        super().__init__()
        self.constraint = constraint
        self.threshold = threshold
        self.max_steps = max_steps

    def combine(self, x0, x, c):
        return (1 - c)*x0 + c*x

    def forward(self, x0, x):
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            one_x0 = x0[i].unsqueeze(0)
            one_x = x[i].unsqueeze(0)
            try:
                if one_x.isnan().any() or one_x0.isnan().any():
                    raise ConvergenceError('All nans')
                f = lambda c: self.constraint(self.combine(one_x0, one_x, c), subset=[i])                
                c = self.binary_search_negative(f, 0, 1, self.threshold, self.max_steps)
                projected = self.combine(one_x0, one_x, c)
            except ConvergenceError as e:
                print(e)
                projected = torch.full_like(one_x, float('nan'))
            results.append(projected)
        return torch.cat(results)

    @staticmethod
    def binary_search_negative(f, a, b, threshold=1e-6, max_steps=100):
        '''
        Binary search between a and b returning c with f(c) with -threshold <= f(c) < 0.
        '''
        sign_a = torch.sign(f(a))
        sign_b = torch.sign(f(b))
        
        if (sign_a < 0) and (sign_b < 0):
            return b
        elif (sign_a > 0) and (sign_b > 0):
            raise ConvergenceError('Binary search has wrong initializaion')
        else:
            for _ in range(max_steps):
                c = (a + b) / 2
                c_value = f(c)
                if (c_value >= -threshold) and (c_value < 0):
                    return c
                else:
                    if torch.sign(c_value) == sign_a:
                        a = c
                    else:
                        b = c
            raise ConvergenceError('Binary search failed to converge')
