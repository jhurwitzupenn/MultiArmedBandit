from __future__ import division
import numpy as np
from strategy import Strategy


class EpsilonGreedyStrategy(Strategy):
    """docstring for EpsilonGreedyAlgorithm"""

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


class SoftMaxStrategy(Strategy):
    """docstring for SoftMaxStrategy"""

    def __init__(self, temperature):
        self._temperature = temperature

    def run(self, T, bandit):
        plays = np.zeros(T)
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        cumProbability = np.zeros(bandit.K)
        divisor = bandit.K
        for t in range(T):
            numerators = np.exp(sampleMeans / self._temperature)
            cumProbability = np.divide(numerators, divisor)

            p = np.random.uniform(0, 1)
            for i in range(bandit.K):
                p = p - cumProbability[i]
                if (p <= 0):
                    index = i

            divisor = sum(numerators)
            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return plays, payoutPerStep, np.sum(payoutPerStep)


class EpsilonFirstStrategy(Strategy):
    """docstring for EpsilonFirstStrategy"""

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def run(self, T, bandit):
        plays = np.zeros(T)
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        exploreRounds = np.ceil(T * self._epsilon)
        index = 0
        for t in range(T):
            if (t < exploreRounds):
                index = bandit.getRandomIndex()

            if (t == exploreRounds - 1):
                index = np.argmax(sampleMeans)

            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return plays, payoutPerStep, np.sum(payoutPerStep)
