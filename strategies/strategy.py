from abc import ABCMeta, abstractmethod


class Strategy:
    """
    Description:
       The abstract class for all algorithms that solve the Bandit problem

    Parameters:
        T: Number of Rounds
        bandit: The bandit that the algorithm will be run on
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    # Returns a tuple of sequence of plays, payout per step, and total payout
    def run(self, T, bandit):
        pass
