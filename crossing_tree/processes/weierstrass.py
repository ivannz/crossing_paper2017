# -*- coding: UTF-8 -*-
from math import log
import numpy as np

from sklearn.base import BaseEstimator as BaseGenerator
from sklearn.utils import check_random_state

class WeierstrassFunction(BaseGenerator):
    """A derived class to produce sample paths of a random Weierstrass function over
    :math:`[0,1]` with a specified Holder exponent.
    """
    def __init__(self, N, lambda_0=1.2, holder=0.5, random_state=None):
        self.N = N
        self.lambda_0 = lambda_0
        self.holder = holder
        self.random_state = random_state
        
    def start(self):
        if hasattr(self, "initialized_") and self.initialized_:
            return

        self.n_layers_ = int(log(self.N * 0.5) / log(self.lambda_0)) + 1

        # Set the random state
        self.random_state_ = check_random_state(self.random_state)
        self.initialized_ = True

    def finish(self):
        if hasattr(self, "initialized_") and self.initialized_:
            self.initialized_ = False

    def draw(self):
        """This implementation constructs a finite-sum apprximation to the
        random-pahse Weierstrass function. The sample points :math:`(t_k)_{i=0}^n\\in [0,1]`
        are such that :math:`0 = t_0 < t_1 < \ldots < t_{n-1} < t_n = 1`, i.e.
        are on a 1D grid with spacing :math:`\\frac{1}{n}`. The :math:`W_H(t_i)`,
        :math:`H\\in (0, 1)`, is approximated with :math:`W_H(t) = \\sum_{k=0}^{M} \\lambda_0^{-k H} W_k(t)`,
        where 
        .. math ::
            W_k(t) = \\cos(2 \\pi \\lambda_0^k t + \\phi_k) \,,
        
        with :math:`M = \\bigl\\lfloor\\frac{\\log \\frac{1}{2} n}{\\log \\lambda_0} \\bigr\\rfloor + 1`,
        which governs the fidelity of the approximation, and is derived from the Nyquist
        frequency. The values :math:`(\\phi_k)_{k=0}^M\\sim \\mathbb{U}[0,2\\pi]` are
        random phase shifts of each layer of harmonics.
        """
        phi_k = self.random_state_.uniform(0, 2 * np.pi, size=self.n_layers_+1)

        t, w = np.linspace(0, 1, num=self.N + 1), np.zeros(self.N + 1, dtype=np.float)

        lambda_k = 1.0
        for k in xrange(self.n_layers_ + 1):
            w += np.cos(2 * np.pi * lambda_k * t + phi_k[k]) * (lambda_k ** -self.holder)
            # spread (self.lambda_0 ** k) exponentiation across layers
            lambda_k *= self.lambda_0

        return t, w - w[0]
