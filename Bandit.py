"""
Bandit Class defines a Multi-Armed bandit
"""

import numpy as np


class Bandit(object):
    """docstring for Bandit"""
    def __init__(self, dists):
        self.arms = _vArm(dists)


class _Arm(object):
    """docstring for _Arm"""
    def __init__(self, dist):
        self.dist = dist



_vArm = np.vectorize(_Arm)
