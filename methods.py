import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from projections import ProjectionLinf
from projections import ProjectionBinarySearch
from objectives import Objective
from constraints import ConstraintMisclassify
from geomloss import SamplesLoss
import schedulers
import distances

def calculate_batch(variables):
    '''
    Simulate batch computing - run decorated function in loop.
    Variables argument: number of variables needed to split for the loop.
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            args_batch = args[:variables]
            args_other = args[variables:]
            batch_size = args_batch[0].shape[0]

            data = []
            for i in range(batch_size):
                data.append(func(*[var[[i]] for var in args_batch], *args_other, **kwargs))
            return torch.cat(data)
        return wrapper
    return decorator


def calculate_gradients(objective, x, norm=None):
    '''
    Calculates gradient of objective(x) at x.
    '''
    x = x.detach()
    x.requires_grad = True
    cost = objective(x)
    grad = torch.autograd.grad(cost, x, torch.ones_like(cost))[0]

    if norm is None:
        return grad
    elif norm == 'sign':
        return grad.sign()
    elif norm == 'l2':
        return grad / grad.norm(p=2, dim=tuple(range(1, x.ndim)), keepdim=True)
    else:
        raise NotImplementedError('Unknown normalization method')


def fgsm(x0, y, classifier, epsilon):
    '''
    Fast Gradient Sign method of Goodfellow.
    '''
    objective = Objective(y, nn.CrossEntropyLoss(), classifier, targeted=False)
    projection = ProjectionLinf(epsilon)
    # TODO I believe that this is wrong. The step_size should be huge.
    # TODO It will not work for epsilon > 1 (but we probably do not need it).
    return projected_gd(x0, objective, projection, grad_norm='sign', steps=1, step_size=1.0, clip=(0, 1))


def projected_gd(x0, objective, projection, grad_norm, step_size, steps=50, clip=None):
    '''
    Projected Gradient Descent method.
    '''
    x = x0.clone()
    for _ in range(steps):

        # Gradient descent step
        grad = calculate_gradients(objective, x, grad_norm)
        x = x - step_size*grad

        # Projection step
        x = projection(x0, x)

        # Stay within pixel range
        if clip is not None:
            x = x.clip(*clip) 
    return x


def bisection_method(x0, x, model, threshold=1e-6):
    '''
    Interpolation between x and x0.

    Assumes that x and x0 are classified into different classes.
    '''
    constraint = ConstraintMisclassify(x0, model)
    projection = ProjectionBinarySearch(constraint, threshold=threshold)
    return projection(x0, x)


@calculate_batch(variables=2)
def projection_method(x0, x, distance, model, xi_c, xi_o, grad_norm_o='l2', grad_norm_c='l2', iters=50):
    '''
    Similar to the HopSkip method, but with any distance metric as an objective.

    TODO better description
    
    TODO why is it done for each channel separately? -> @calculate_batch(variables=2)

    TODO:
        - Do we even need to do the initial projection?
        - Do we need to normalize gradient in the projection step?
    '''
    constraint = ConstraintMisclassify(x0, model)
    projection = ProjectionBinarySearch(constraint, threshold=0.001)
    objective = lambda x: distance(x0, x)

    # Project in direction of min distance
    grad_objective = calculate_gradients(objective, x, norm=grad_norm_o)
    x = projection(x, x - grad_objective)

    # TODO chceme ty iterace takto?
    for t in range(iters):
        # Step in direction of constraint
        grad_constraint = calculate_gradients(constraint, x, norm=grad_norm_c)
        x = x - xi_c(t)*grad_constraint

        # Project in direction of min distance
        grad_objective = calculate_gradients(objective, x, norm=grad_norm_o)
        x = projection(x, x - xi_o(t)*grad_objective)
    return x.detach()


@calculate_batch(variables=1)
def penalty_method(x0, distance, model, xi, rho, grad_norm='l2', iters=100, max_unchanged=10):
    '''
    Penalty method with distance as objective and misclassification constraint. 
    Penalization: rho * max(g(x), 0)^2, where rho is a penalization parameter
    '''
    constraint = ConstraintMisclassify(x0, model)
    result = torch.full_like(x0, float('nan'))

    # TODO this seems to be wrong. where is the initialization?

    # TODO check it later again

    x = x0.clone()
    unchanged = 0
    for t in range(iters):
        # Optimization step
        l = lambda x: distance(x0, x) + rho(t)*F.relu(constraint(x))**2
        grad = calculate_gradients(l, x, norm=grad_norm)
        x = x - xi(t)*grad

        # Termination condition
        if torch.sign(constraint(x)) > 0:
            unchanged += 1
        else:
            unchanged = 0
            result = x
        if unchanged > max_unchanged:
            break
    return result.detach()

@calculate_batch(variables=2)
def projected_gd_method(x0, x, distance, model, xi_o, xi_c, iters=100):
    # TODO what is the difference from projection_method?

    # TODO check it later
    constraint = ConstraintMisclassify(x0, model)
    projection = ProjectionBinarySearch(constraint, threshold=0.001)
    x = projection(x0, x)
    for t in range(iters):
        # Update objective
        objective = lambda x: distance(x0, x)
        grad_objective = calculate_gradients(objective, x, norm='l2')
        x = x - xi_o(t)*grad_objective

        # Project to feasible region
        grad_constraint = calculate_gradients(constraint, x, norm='l2')
        x = x - xi_c(t)*grad_constraint
        x = projection(x0, x)
    return x.detach()


@calculate_batch(variables=2)
def hopskip_method(x0, x, model, grad_norm='l2', iters=50):
    '''
    Method based on hopskip paper - optimizes squared l2 distance between x0 and x.
    '''
    # TODO are we still using this?
    constraint = ConstraintMisclassify(x0, model)
    projection = ProjectionBinarySearch(constraint, threshold=0.001, inside=False)
    x = projection(x0, x)

    for t in range(iters):
        grad_constraint = calculate_gradients(constraint, x, norm=grad_norm)

        # Xi selection
        xi = torch.norm(x - x0, p=2) / np.sqrt(t+1)
        while constraint(x - xi*grad_constraint) > 0:
            xi = xi / 2

        # Update and project
        x = x - xi*grad_constraint
        x = projection(x0, x)
    return x.detach()

# TODO what are the wrapper methods good for?
###### Wrapped method
def penalty_method_wrapped(
    x0,
    x,
    classifier,
    generator,
    distance = 'l2',
    distance_args = None,
    iters = 100,
    grad_norm = 'l2',
    rho = None,
    xi = None,
):

    if rho is None:
        rho = {'scheduler': 'SchedulerStep', 'initial': 10e8, 'gamma': 1, 'n': 10}
    if xi is None:
        xi = {'scheduler': 'SchedulerExponential', 'initial': 1, 'gamma': 0.01}
    rho = getattr(schedulers, rho.pop('scheduler'))(**rho)
    xi = getattr(schedulers, xi.pop('scheduler'))(**xi)

    if distance == 'l2':
        transform = distances.Decoded(generator)
        distance_function = distances.L2(transform)

    if distance == 'wd':
        transform = distances.DecodedDistribution(generator)
        distance_function = distances.GeomLoss(SamplesLoss(**distance_args), transform)

    return penalty_method(x0, distance_function, classifier, xi, rho, grad_norm, iters)


def projection_method_wrapped(
    x0,
    x,
    classifier,
    generator,
    distance = 'l2',
    distance_args = None,
    xi_c = None,
    xi_o = None,
    grad_norm_o = 'l2',
    grad_norm_c = 'l2',
    iters = 50
):
    if xi_c is None:
        xi_c = {'scheduler': 'SchedulerPower', 'initial': 1, 'power': -0.5}
    if xi_o is None:
        xi_o = {'scheduler': 'SchedulerConstant','alpha': 1}

    xi_c = getattr(schedulers, xi_c.pop('scheduler'))(**xi_c)
    xi_o = getattr(schedulers, xi_o.pop('scheduler'))(**xi_o)

    if distance == 'l2':
        transform = distances.Decoded(generator)
        distance_function = distances.L2(transform)

    if distance == 'wd':
        transform = distances.DecodedDistribution(generator)
        distance_function = distances.GeomLoss(SamplesLoss(**distance_args), transform)

    return projection_method(x0, x, distance_function, classifier, xi_c, xi_o, grad_norm_o, grad_norm_c, iters)
