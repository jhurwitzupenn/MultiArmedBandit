"""
Bandit Class defines a Multi-Armed bandit.
"""
import numpy as np
from distributions import Distribution


class _Arm(object):
    """
    Description:
        The _Arm class is a wrapper for a distribution and represents the 
        lever of a slot machine

    Parameters:
        dist : The distribution that is wrapped by the arm
    """

    def __init__(self, dist):
        self._dist = dist

    def payout(self):
        return self._dist.getSample()

    def getMean(self):
        return self._dist.getMean()


_vArm = np.vectorize(_Arm)


class Bandit(object):
    """
    Description:
        The Bandit class has multiple arms that each contain a distribution.
        The User interacts with the bandit class and can pull predetermined or
        random levers. The Bandit keeps track of the optimal strategy and can 
        return the regret of a lever pull.

    Parameters:
        dists : The distributions that correspond to each arm of the bandit
    """

    def __init__(self, dists):
        for dist in dists:
            if not isinstance(dist, Distribution):
                raise TypeError("Must pass in Distribution Iterable")

        self._arms = _vArm(dists)
        self.K = self._arms.size
        self._optimalMean = max(dists, key=lambda d: d.getMean()).getMean()

    def getPayout(self, armIndex):
        if (armIndex >= self.K):
            raise IndexError("Arm index is out of bounds")
        return self._arms[armIndex].payout()

    def getRandomIndex(self):
        return np.random.randint(0, self.K)

    def getOptimalStrategyPayout(self, T):
        return T * self._optimalMean

    def getRandomArm(self):
        index = np.random.randint(0, self.K)
        return index, self._arms[index]

    def getRegret(self, plays):
        steps = plays.size
        regretPerStep = np.zeros(steps)
        totalRegret = 0
        for t in range(steps):
            choice = plays[t]
            regretPerStep[t] = np.subtract(self._optimalMean,
                                           self._arms[int(choice)].getMean())
            totalRegret += regretPerStep[t]
        return regretPerStep, totalRegret
