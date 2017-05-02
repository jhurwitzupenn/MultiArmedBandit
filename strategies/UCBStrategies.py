import numpy as np
from Strategy import Strategy


class UCB1Strategy(Strategy):
    """docstring for EpsilonGreedyAlgorithm"""

    def run(self, T, bandit):
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        for t in range(T):
            if (t < bandit.K):
                index = t
            else:
                summands = np.sqrt(np.divide((2 * np.log(t)), cumPlays))
                scores = sampleMeans + summands
                index = np.argmax(scores)

            payout = bandit.getPayout(index);
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return payoutPerStep
