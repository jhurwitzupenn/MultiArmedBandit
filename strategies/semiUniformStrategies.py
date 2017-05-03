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


class ExploreThenCommit(Strategy):
    """docstring for ExploreThenExploit"""

    def __init__(self, exploreDepth):
        self._exploreDepth = exploreDepth

    def run(self, T, bandit):
        plays = np.zeros(T)
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        for t in range(T):
            if (t < self._exploreDepth * bandit.K):
                index = t % bandit.K

            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])
            if (t == self._exploreDepth * bandit.K - 1):
                index = np.argmax(sampleMeans)

        return plays, payoutPerStep, np.sum(payoutPerStep)
