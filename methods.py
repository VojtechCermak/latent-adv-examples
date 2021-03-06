import torch
import torch.nn as nn
import torch.nn.functional as F
from projections import ConvergenceError, ProjectionLinf
from projections import ProjectionBinarySearch
from objectives import Objective
from constraints import ConstraintMisclassify, ConstraintClassifyTarget
from geomloss import SamplesLoss
import schedulers
import distances


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


class BaseMethod():
    def __init__(self, distance, distance_args, constraint):
        if distance_args is None:
            distance_args = {}
        self.distance = distance
        self.distance_args = distance_args
        self.constraint = constraint

    def method(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, x0, x_init, classifier, generator, target):
        distance = self.parse_distance(generator)

        batch_size = x0.shape[0]
        data = []
        for i in range(batch_size):
            objective = lambda x: distance(x0[[i]], x)
            constraint = self.parse_constraint(classifier, x0, target, subset=[i])
            result = self.method(objective, constraint, x_init = x_init[[i]], **self.params)
            data.append(result)
        return torch.cat(data)

    def __repr__(self):
        return str(self.__class__.__name__)

    def parse_distance(self, generator):
        '''
        Parse distance function based on self.distance and self.distance_args.
        '''
        if self.distance == 'l2':
            if generator is None:
                distance = distances.L2()
            else:
                transform = distances.Decoded(generator)
                distance = distances.L2(transform)

        elif self.distance == 'wd':
            if generator is None:
                distance = distances.GeomLoss(SamplesLoss(**self.distance_args))
            else:
                transform = distances.DecodedDistribution(generator)
                distance = distances.GeomLoss(SamplesLoss(**self.distance_args), transform)

        else:
            raise ValueError('Invalid distance.')
        return distance

    def parse_constraint(self, classifier, x0, target, subset=None):
        '''
        Parse constraint function based on self.constraint.
        '''
        if subset is None:
            subset = slice(None)

        if self.constraint == 'misclassify':
            constraint = ConstraintMisclassify(x0[subset], classifier)

        elif self.constraint == 'targeted':
            if target is None:
                raise ValueError("Targeted attack should have valid target.")
            constraint = ConstraintClassifyTarget(target[subset], classifier)

        else:
            raise ValueError('Invalid constraint.')
        return constraint

    def parse_scheduler(self, dictionary):
        scheduler_class = getattr(schedulers, dictionary['scheduler'])
        scheduler = scheduler_class(**dictionary['params'])
        return scheduler


class PenaltyMethod(BaseMethod):
    def __init__(
        self,
        distance = 'l2',
        distance_args = None,
        constraint = 'misclassify',
        rho = None,
        xi  = None,
        iters = 100,
        grad_norm = 'l2',
    ):
        super().__init__(distance, distance_args, constraint)
        if rho is None:
            raise ValueError('Invalid rho')
        if xi is None:
            xi  = {'scheduler': 'SchedulerExponential', 'params':{'initial': 1, 'gamma': 0.01 }}

        self.params_all = {str(k): str(v) for k, v in locals().items() if k not in ["__class__"]}
        self.params = {
            'rho': self.parse_scheduler(rho),
            'xi' : self.parse_scheduler(xi),
            'iters': iters,
            'grad_norm': grad_norm,
        }

    def __call__(self, x0, classifier, generator, target=None, **kwargs):
        x_init = x0
        return super().__call__(x0, x_init, classifier, generator, target)

    def method(self, objective, constraint, x_init, xi, rho, grad_norm='l2', iters=100):
        '''
        Penalty method args: objective, constraint, x_init, hyperpars
        Penalization: rho * max(g(x), 0)^2, where rho is a penalization parameter

        There is no guarantee that the final point is feasible.
        '''
        x = x_init.clone()

        for t in range(iters):
            l = lambda x: objective(x) + rho(t)*F.relu(constraint(x))**2
            grad = calculate_gradients(l, x, norm=grad_norm)
            x = x - xi(t)*grad
        return x.detach()


class ProjectionMethod(BaseMethod):
    def __init__(
        self,
        distance = 'l2',
        distance_args = None,
        constraint = 'misclassify',
        xi_c = None,
        xi_o = None,
        iters = 100,
        grad_norm_o = 'l2',
        grad_norm_c = 'l2',
        threshold = 1e-3,
    ):
        super().__init__(distance, distance_args, constraint)
        if xi_c is None:
            xi_c = {'scheduler': 'SchedulerPower', 'params': {'initial': 1, 'power': -0.5}}
        if xi_o is None:
            xi_o = {'scheduler': 'SchedulerConstant', 'params': {'alpha': 1}}

        self.params_all = {str(k): str(v) for k, v in locals().items() if k not in ["__class__"]}
        self.params = {
            'xi_c': self.parse_scheduler(xi_c),
            'xi_o': self.parse_scheduler(xi_o),
            'iters': iters,
            'grad_norm_o': grad_norm_o,
            'grad_norm_c': grad_norm_c,
            'threshold': threshold,
        }

    def __call__(self, x0, x_init, classifier, generator, target=None, **kwargs):
        return super().__call__(x0, x_init, classifier, generator, target)

    def method(self, objective, constraint, x_init, xi_c, xi_o, grad_norm_o='l2', grad_norm_c='l2', iters=50, threshold=1e-3):
        '''
        Projected gradient method. The initial point x_init must be feasible constraint(x_init) <= 0.

        It alternates between bouncing off the boundary, decreasing the objective and potential projecting
        unfeasible points to the feasibile region. All iterations are feasible (unless the projection fails).
        '''
        x = x_init.clone()
        projection = ProjectionBinarySearch(constraint, threshold=threshold)

        # Project in direction of min distance
        grad_objective = calculate_gradients(objective, x, norm=grad_norm_o)
        x_next = x - xi_o(0)*grad_objective
        x = projection(x, x_next)

        for t in range(iters):
            # Step in direction of constraint
            if not (x_next == x).all().item():
                grad_constraint = calculate_gradients(constraint, x, norm=grad_norm_c)

                lr = xi_c(t)
                converged = False
                for _ in range(100):
                    if constraint(x - lr*grad_constraint) < 0 :
                        converged = True
                        break
                    else:
                        lr = lr / 2
                if not converged:
                    raise ConvergenceError("Step correction is in infinite cycle")
                x = x - lr*grad_constraint

            # Project in the direction of objective
            grad_objective = calculate_gradients(objective, x, norm=grad_norm_o)
            x_next = x - xi_o(0)*grad_objective
            x = projection(x, x_next)
        return x.detach()

