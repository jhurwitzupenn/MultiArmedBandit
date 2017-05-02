"""
Bandit Class defines a Multi-Armed bandit.
"""
import numpy as np
from Distributions import Distribution


class _Arm(object):
    """docstring for _Arm"""
    def __init__(self, dist):
        self._dist = dist

    def payout(self):
        return self._dist.getSample()


_vArm = np.vectorize(_Arm)


class Bandit(object):
    """docstring for Bandit"""
    def __init__(self, dists):
        for dist in dists:
            if not isinstance(dist, Distribution):
                raise TypeError("Must pass in Distribution Iterable")

        self.arms = _vArm(dists)
        self.K = self.arms.size
        self._optimalMean = max(dists, key=lambda d: d.getMean()).getMean()

    def getOptimalStrategyPayout(self, T):
        return T * self._optimalMean

    def getRandomArm(self):
        index = np.random.randint(0, self.K)
        return index, self.arms[index]
