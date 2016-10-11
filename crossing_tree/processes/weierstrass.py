# -*- coding: UTF-8 -*-
"""A module with the Weierstrass function generator.
"""
from math import log
import numpy as np

from sklearn.base import BaseEstimator as BaseGenerator
from sklearn.utils import check_random_state

class WeierstrassFunction(BaseGenerator):
    """A class to produce sample paths of a random Weierstrass function over :math:`[0,1]`
    with a specified Holder exponent and fundamental harmonic :math:`\\lambda_0`.

    Details
    -------
    This implementation constructs a finite-sum approximation to the random-phase
    Weierstrass function :math:`W_H(t)`, :math:`H\\in (0, 1)`, sampled over 1d mesh
    of :math:`n+1` points with spacing :math:`\\frac{1}{n}`.

    The function is defined by the series :math:`\\sum_{k\\in\\mathbb{Z}} W_k(t)`
    with
    .. math ::
        W_k(t) =  \\lambda_0^{-k H} \\bigl(
                \\cos(2 \\pi \\lambda_0^k t + \\phi_k) - \\cos(\\phi_k)
            \\bigr) \\,,
    where :math:`(\\phi_k)_{k\\in\\mathbb{Z}} \\sim \\mathbb{U}[0,2\\pi]` are iid
    random phase shifts of each layer of harmonics. The function is approximated
    by a truncated sum :math:`\\sum_{k=-M}^{M} W_k(t)` where 
    .. math ::
        M = \\bigl\\lfloor
                \\frac{\\log \\frac{1}{2} n}{\\log \\lambda_0}
            \\bigr\\rfloor + 1 \\,,
    which governs the fidelity of the approximation.

    One-sided approximations drops the negative index series altogether.
    """
    def __init__(self, N, lambda_0=1.2, holder=0.5, random_state=None, one_sided=False):
        self.N = N
        self.lambda_0 = lambda_0
        self.holder = holder
        self.random_state = random_state
        self.one_sided = one_sided
        
    def start(self):
        """Initialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            return

        self.n_layers_ = int(log(self.N * 0.5) / log(self.lambda_0)) + 1

        # Set the random state
        self.random_state_ = check_random_state(self.random_state)
        self.initialized_ = True

    def finish(self):
        """Deinitialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            self.initialized_ = False

    def draw(self):
        """Draw a single realisation of the processes trajectory from the generator.
        """
        if self.one_sided:
            return self._onesided()
        else:
            return self._twosided()

    def _onesided(self):
        if not(hasattr(self, "initialized_") and self.initialized_):
            raise RuntimeError("""The generator has not been initialized properly. """
                               """Please call `.start()` before calling `.draw()`.""")

        phi_k = self.random_state_.uniform(0, 2 * np.pi, size=self.n_layers_ + 1)

        t, w = np.linspace(0, 1, num=self.N + 1), np.zeros(self.N + 1, dtype=np.float)

        lambda_k = 1.0
        for k in xrange(self.n_layers_ + 1):
            w += np.cos(2 * np.pi * lambda_k * t + phi_k[k]) * (lambda_k ** -self.holder)
            # spread (self.lambda_0 ** k) exponentiation across layers
            lambda_k *= self.lambda_0

        # No need to sibtract w_0 inside the loop, since it cannot potentially overflow.
        return t, w - w[0]

    def _twosided(self):
        if not(hasattr(self, "initialized_") and self.initialized_):
            raise RuntimeError("""The generator has not been initialized properly. """
                               """Please call `.start()` before calling `.draw()`.""")

        # \phi_0, [\phi_1, \ldots, \phi_M], [\phi_{-M}, \ldots, \phi_{-1}]
        phi_k = self.random_state_.uniform(0, 2 * np.pi, size=2 * self.n_layers_ + 1)
        cosphi_k = np.cos(phi_k)

        t = np.linspace(0, 1, num=self.N + 1)
        w = np.cos(2 * np.pi * t + phi_k[0]) - cosphi_k[0]

        lambda_k = self.lambda_0
        for k in xrange(1, self.n_layers_ + 1):
            # to avoid overflow subtract parts of w_0 inside the loop.
            w += (np.cos(2 * np.pi * lambda_k * t + phi_k[k])
                  - cosphi_k[k]) * (lambda_k ** -self.holder)

            w += (np.cos(2 * np.pi * t / lambda_k + phi_k[-k])
                  - cosphi_k[-k]) * (lambda_k ** self.holder)

            lambda_k *= self.lambda_0

        return t, w
