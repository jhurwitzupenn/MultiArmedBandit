import numpy as np
import math as math
from strategy import Strategy


class UCB1Strategy(Strategy):
    """
    Description:
        UCB1Strategy picks arms based on sample means and the estimated size of
        the confidence interval of average rewards. No parameters taken in.
    """

    def run(self, T, bandit):
        plays = np.zeros(T)
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

            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return plays, payoutPerStep, np.sum(payoutPerStep)


class UCB2Strategy(Strategy):
    """
    Description:
        UCB2Strategy picks arms based on sample means and the estimated size of
        the confidence interval of average rewards.
        
        Parameters:
            alpha: Regulates the width of the confidence intervals
    """

    def __init__(self, alpha):
        self._alpha = alpha

    def tau(self, r):
        return np.ceil((1 + self._alpha)**r)

    def summand(self, n, r):
        dividend = (1 + self._alpha) * np.log(math.e * n / self.tau(r))
        divisor = 2 * self.tau(r)
        return np.sqrt(np.divide(dividend, divisor))

    def run(self, T, bandit):
        plays = np.zeros(T)
        payoutPerStep = np.zeros(T)
        cumPayouts = np.zeros(bandit.K)
        cumPlays = np.zeros(bandit.K)
        sampleMeans = np.zeros(bandit.K)
        epochs = np.zeros(bandit.K)

        for index in range(bandit.K):
            plays[index] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[index] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])
            epochs[index] += 1

        t = bandit.K - 1

        while(t < T):
            curSummands = self.summand(t, epochs)
            scores = sampleMeans + curSummands
            index = np.argmax(scores)
            curEpochs = epochs[index]
            curPlays = self.tau(curEpochs + 1) - self.tau(curEpochs)
            for i in range(int(curPlays)):
                payout = bandit.getPayout(index)
                cumPayouts[index] += payout
                cumPlays[index] += 1
                payoutPerStep[t] = payout
                sampleMeans[index] = np.divide(cumPayouts[index],
                                               cumPlays[index])
                plays[t] = index
                t += 1
                if (t >= T):
                    break
            epochs[index] += 1

        return plays, payoutPerStep, np.sum(payoutPerStep)
