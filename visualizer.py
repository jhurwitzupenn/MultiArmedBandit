import matplotlib.pyplot as plt
import numpy as np


class Visualizer(object):

    def __init__(self, algorithms, labels=None):
        if (labels):
            if (len(labels) != len(algorithms)):
                raise ValueError("labels must be same length as algorithms")
        self._algorithms = algorithms
        self._labels = labels

    def graphCumulativePayouts(self, bandit, T=1000, showOptimal=False):
        x = range(T)
        for algorithm in self._algorithms:
            plt.plot(x, np.cumsum(algorithm.run(T, bandit)[1]))

        if (showOptimal):
            plt.plot(x, np.linspace(0, bandit.getOptimalStrategyPayout(T), T))

        if (self._labels):
            plt.legend(self._labels + ['optimal'])

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Payout')

        return plt

    def graphCumulativeRegrets(self, bandit, T=1000):
        x = range(T)
        for algorithm in self._algorithms:
            plays, _, _ = algorithm.run(T, bandit)
            plt.plot(x, np.cumsum(bandit.getRegrets(plays)))

        if (self._labels):
            plt.legend(self._labels)

        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Regret')

        return plt
