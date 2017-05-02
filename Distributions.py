import numpy as np
from abc import ABCMeta, abstractmethod
from functools import partial


class Distribution:
    __metaclass__ = ABCMeta

    @abstractmethod
    # Returns a single sample (int) drawn from the distribution
    def getSample(self):
        pass

    @abstractmethod
    # Returns the mean (float) of the distribution
    def getMean(self):
        pass


class Binomial(Distribution):
    """docstring for _Binomial"""
    def __init__(self, n, p):
        self._mean = n * p
        self._binom = partial(np.random.binomial, n=n, p=p)

    def getSample(self):
        return self._binom()

    def getMean(self):
        return self._mean
