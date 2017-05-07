import matplotlib.pyplot as plt
import numpy as np


class Visualizer(object):

    def __init__(self, algorithms, labels):
        if (labels):
            if (len(labels) != len(algorithms)):
                raise ValueError("labels must be same length as algorithms")
        self._algorithms = algorithms
        self._labels = labels
        self._isRun = False

    def run(self, T, bandit):
        self._isRun = True
        self._bandit = bandit
        self._T = T
        self._playsSequences = []
        self._payoutSequences = []
        for algorithm in self._algorithms:
            plays, payoutPerStep, _ = algorithm.run(T, bandit)
            self._playsSequences.append(plays)
            self._payoutSequences.append(payoutPerStep)

    def graphCumulativePayouts(self, showOptimal=False):
        if not (self._isRun):
            raise ValueError("Run the visualizer first")

        x = range(self._T)
        for payout in self._payoutSequences:
            plt.plot(x, np.cumsum(payout))

        if (showOptimal):
            optimalPayout = self._bandit.getOptimalStrategyPayout(self._T)
            plt.plot(x, np.linspace(0, optimalPayout, self._T))
            plt.legend(self._labels + ['optimal'])
        else:
            plt.legend(self._labels)

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Payout')

        return plt

    def graphCumulativeRegrets(self):
        if not (self._isRun):
            raise ValueError("Run the visualizer first")

        for plays in self._playsSequences:
            plt.plot(range(self._T),
                     np.cumsum(self._bandit.getRegret(plays)[0]))

        plt.legend(self._labels)

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Regret')

        return plt

    def graphPerStepRegret(self):
        if not (self._isRun):
            raise ValueError("Run the visualizer first")

        for plays in self._playsSequences:
            regretPerStep, _ = self._bandit.getRegret(plays)
            windowSize = int(plays.size / 10)
            smoothedRegretPerStep = movingAverage(regretPerStep, windowSize)
            plt.plot(smoothedRegretPerStep[windowSize + 1:])

        plt.legend(self._labels)

        plt.xlabel('Iterations')
        plt.ylabel('Per Step Regret (' + str(int(windowSize)) + '-windowed)')

        return plt


def movingAverage(interval, window_size):
    window = np.ones(window_size) / float(window_size)
    return np.convolve(interval, window, 'same')
