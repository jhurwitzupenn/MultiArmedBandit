from __future__ import division
import numpy as np
from strategy import Strategy


class EpsilonGreedyStrategy(Strategy):
    """
    Description:
        The EpsilonGreedyStrategy picks a lever uniformly at random with 
        probability epsilon, and picks the level with the largest sample mean 
        with probability 1 - epsilon.

    Parameters:
        epsilon : probablity value
    """

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def run(self, T, bandit):
        plays = np.zeros(T)
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        for t in range(T):
            if (np.random.random() > 1 - self._epsilon):
                index = bandit.getRandomIndex()
            else:
                index = np.argmax(sampleMeans)

            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return plays, payoutPerStep, np.sum(payoutPerStep)
        

class EpsilonFirstStrategy(Strategy):
    """
    Description:
        The EpsilonGreedyStrategy explores for epislon*number of rounds and
        then chooses the lever with the highest sample mean to exploit for the
        rest of the rounds.

    Parameters:
        epsilon : percentage of explore rounds
    """

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def run(self, T, bandit):
        plays = np.zeros(T)
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        exploreRounds = T * self._epsilon
        for t in range(T):
            if (t < exploreRounds):
                index = bandit.getRandomIndex()

            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])
            if (t == exploreRounds - 1):
                index = np.argmax(sampleMeans)

        return plays, payoutPerStep, np.sum(payoutPerStep)