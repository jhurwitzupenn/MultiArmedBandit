import numpy as np
from Strategy import Strategy


class EpsilonGreedyStrategy(Strategy):
    """docstring for EpsilonGreedyAlgorithm"""
    
    def __init__(self, epsilon):
        self._epsilon = epsilon

    def run(self, T, bandit):
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        for t in range(T):
            if (np.random.random() > 1 - self._epsilon):
                index = bandit.getRandomIndex()
            else:
                index = np.argmax(sampleMeans)

            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return np.sum(payoutPerStep), payoutPerStep