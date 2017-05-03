from abc import ABCMeta, abstractmethod


class Strategy:
    __metaclass__ = ABCMeta

    @abstractmethod
    # Returns a tuple of sequence of plays, payout per step, and total payout
    def run(self, T, bandit):
        pass
