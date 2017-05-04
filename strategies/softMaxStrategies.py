import numpy as np
from strategy import Strategy


class SoftMaxStrategyFixed(Strategy):
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
            numerators = np.exp(sampleMeans/self._temperature)
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

