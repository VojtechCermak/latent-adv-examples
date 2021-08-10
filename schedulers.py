import numpy as np
from inspect import signature

class Scheduler():
    '''
    Base class for learning rate schedulers. Override the step method.
    '''
    def __repr__(self):
        fields = tuple(f'{k}={v}' for k,v  in self.__dict__.items() if k in signature(self.__class__).parameters)
        return f"{self.__class__.__name__}({', '.join(fields)})"


class SchedulerConstant(Scheduler):
    '''
    Scheduling steps at constant learning rate alpha.
    '''
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, i):
        return self.alpha


class SchedulerPower(Scheduler):
    '''
    Scheduling steps according to: init*(1+i)^power
    '''
    def __init__(self, initial, power):
        self.initial = initial
        self.power = power

    def __call__(self, i):
        return self.initial * (1 + i)**(self.power)


class SchedulerExponential(Scheduler):
    '''
    Scheduling steps according to: init*e^(-i*gamma)
    '''
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma  = gamma

    def __call__(self, i):
        return self.initial * np.exp(-self.gamma*i)


class SchedulerStep(Scheduler):
    '''
    Decrease step every n steps based on gamma.
    '''
    def __init__(self, initial, n, gamma=1):
        self.initial = initial
        self.gamma = gamma
        self.n = n

    def __call__(self, i):
        return self.initial * np.power(2, -self.gamma*np.floor((1/self.n)*i))
