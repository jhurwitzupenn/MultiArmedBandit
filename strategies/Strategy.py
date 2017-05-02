from abc import ABCMeta, abstractmethod


class Strategy:
    __metaclass__ = ABCMeta

    @abstractmethod
    # Returns payouts at each step of the algorithm
    def run(self, T, bandit):
        pass
