from __future__ import division
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
    """
    Description:
       Binomial Distribution class that can get the mean and a random sample
       from a binomial distribution

    Parameters:
        n : Number of trials
        p : Probability of success
    """

    def __init__(self, n, p):
        self._mean = n * p
        self._binom = partial(np.random.binomial, n=n, p=p)

    def getSample(self):
        return self._binom()

    def getMean(self):
        return self._mean


class Gaussian(Distribution):
    """
    Description:
       Gaussian Distribution class that can get the mean and a random sample
       from a gausssian distribution

    Parameters:
        loc : Mean of the distribution
        scale : Variance of the distribution
    """

    def __init__(self, loc, scale):
        self._mean = loc
        self._gaussian = partial(np.random.normal, loc=loc, scale=scale)

    def getSample(self):
        return self._gaussian()

    def getMean(self):
        return self._mean


class Uniform(Distribution):
    """
    Description:
       Uniform Distribution class that can get the mean and a random sample
       from a uniform distribution

    Parameters:
        low : Lower boundry of output interval
        high : Upper boundry of output interval
    """

    def __init__(self, low, high):
        self._mean = (low + high) / 2
        self._uniform = partial(np.random.uniform, low=low, high=high)

    def getSample(self):
        return self._uniform()

    def getMean(self):
        return self._mean
