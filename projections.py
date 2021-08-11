import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvergenceError(Exception):
    pass
    # TODO Shouldn't an error be here?


class Projection():
    '''
    Super class for projections of x onto some ball around x0.
    '''
    def __call__(self, x0, x):
        pass
        # TODO Shouldn't an error be here?


class ProjectionIdentity(Projection):
    '''
    Project x to itself.
    '''
    def __call__(self, x0, x):
        return x


class ProjectionL2(Projection):
    '''
    Project x to the Euclidean epsilon ball around x0.
    '''
    def __init__(self, eps):
        self.eps = eps

    def projection(self, x, eps):
        x_norm = x.norm(p=2, dim=tuple(range(1, x.ndim)), keepdim=True)
        x = torch.where(x_norm > eps, eps*(x / x_norm), x)
        return x

    def __call__(self, x0, x):
        delta = x - x0
        delta_projected = self.projection(delta, self.eps)
        return x0 + delta_projected


class ProjectionLinf(Projection):
    '''
    Project x to the l-inf epsilon ball around x0.
    '''
    def __init__(self, eps):
        self.eps = eps

    def projection(self, x, eps):
        return x.clamp(-eps, eps)

    def __call__(self, x0, x):
        delta = x - x0
        delta_projected = self.projection(delta, self.eps)
        return x0 + delta_projected


class ProjectionBinarySearch(nn.Module):
    '''
    Find x' on line between x0 and x a such that f(x', x0) <= eps.
    Implemented using binary search.
    '''
    def __init__(self, constraint, inside, threshold=0.001, max_steps=100):
        super().__init__()
        self.constraint = constraint
        self.inside = inside
        self.threshold = threshold
        self.max_steps = max_steps

    def combine(self, x0, x, c):
        return (1 - c)*x0 + c*x

    def projection(self, x0, x, function):
        same_sign = torch.sign(function(0)) == torch.sign(function(1))
        if self.inside:
            # x0 and x should be at the same side
            if same_sign:
                return x
            else:
                a, b = self.binary_search(function, 0, 1, self.threshold, self.max_steps)
                return self.combine(x0, x, a)
        else:
            # x0 and x should be at the different side
            if same_sign:
                a, b = self.expand_interval(function, 0, 1)
            else:
                a, b = 0, 1
            a, b = self.binary_search(function, a, b, self.threshold, self.max_steps)
            return self.combine(x0, x, b)

    def forward(self, x0, x):
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            try:
                one_x0 = x0[i].unsqueeze(0)
                one_x = x[i].unsqueeze(0)
                function = lambda c: self.constraint(self.combine(one_x0, one_x, c), subset=[i])
                projected = self.projection(one_x0, one_x, function)
            except ConvergenceError as e:
                print(e)
                projected = torch.full_like(x, float('nan'))
            results.append(projected)
        return torch.cat(results)

    @staticmethod
    def binary_search(f, a, b, threshold=0.001, max_steps=100):
        sign_a = torch.sign(f(a))
        sign_b = torch.sign(f(b))
        assert sign_b != sign_a
        for _ in range(max_steps):
            c = (a + b) / 2
            c_value = f(c)
            if abs(c_value) < threshold:
                # TODO I do not think that this is a good idea. It will give horrible results for any linear function f.
                return a, b
            else:
                if torch.sign(c_value) == sign_a:
                    a = c
                else:
                    b = c
        raise ConvergenceError('Binary search failed to converge')

    @staticmethod
    def expand_interval(f, a, b, max_steps=10):
        assert a != b
        for _ in range(max_steps):
            a_value = f(a)
            b_value = f(b)
            if torch.sign(a_value) == torch.sign(b_value):
                sign = torch.sign(a_value)
                if ((sign > 0) and (a_value > b_value)) or ((sign < 0) and (a_value < b_value)):
                    b = b + 2*abs(a-b)
                else:
                    a = a - 2*abs(a-b)
            else:
                return a, b
        raise ConvergenceError('Expanding interval failed to converge')


