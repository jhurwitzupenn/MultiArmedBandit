import numpy as np
from strategy import Strategy


class SoftMaxStrategyFixed(Strategy):
    """
    Description:
        The SoftMaxStrategFixed the probability of exploring a lever is
        weighted by its sample mean so exploration is only semi-random,
        favoring the exploration of good levers. This algorithm weights the
        probability of choosing an arm by the boltzmann distribution which
        takes depends on the sample mean and a temperature constant.

    Parameters:
        temperature : parameter of the boltzmann function
    """

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
            divisor = sum(numerators)
            cumProbability = np.divide(numerators, divisor)

            p = np.random.uniform(0, 1)
            for i in range(bandit.K):
                p = p - cumProbability[i]
                if (p <= 0):
                    index = i

            plays[t] = index
            payout = bandit.getPayout(index)
            cumPayouts[index] += payout
            cumPlays[index] += 1
            payoutPerStep[t] = payout
            sampleMeans[index] = np.divide(cumPayouts[index], cumPlays[index])

        return plays, payoutPerStep, np.sum(payoutPerStep)
