# -*- coding: UTF-8 -*-
"""A module with the Hermite process generator.
"""
import numpy as np

from numpy.polynomial.hermite_e import hermeval

from sklearn.base import BaseEstimator as BaseGenerator
from .gaussian import FractionalGaussianNoise


class HermiteProcess(BaseGenerator):
    r"""A derived class to produce sample paths of a Hermite process of order
    `degree` with specified Hurst exponent (fractional integration parameter).
    For the best performance `N * n_downsample` should be a power of two. The
    hurst exponent for this process is
    :math:`H = 1 + degree * (H_{\text{fgn}} - 1)`.

    Returns a process sampled on :math:`0.0=t_0<t_1<\ldots<t_N=1.0` with equal
    spacing given by :math:`N^{-1}`.

    Details
    -------
    When downsampling parameter (`n_downsample`) tends to infinity the process,
    converges in distribution to the Rosenblatt process or in general to a
    Hermite process. This stems from the `non-central limit theorem`, i.e.

    .. math :
        Z^k(t) = \frac{1}{n^\alpha}
                \sum_{j=1}^{\lfloor kt\rfloor} H(\xi_j)
            \,,

    converges to :math:`Z_\frac{\alpha}{2}(t)` -- a hermite process. Thus
    increasing `n_downsample` gives better approximation.

    In theory it should tend to infinity. This is a serious drawback.
    c.f. [Abry, Pipiras; 2005]
    """
    def __init__(self, N, degree=2, n_downsample=16, hurst=0.5,
                 random_state=None, n_threads=1):
        self.random_state = random_state
        self.n_threads = n_threads
        self.N = N
        self.hurst = hurst
        self.degree = degree
        self.n_downsample = n_downsample

    def start(self):
        """Initialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            return

        self.fgn_ = FractionalGaussianNoise(
            N=self.n_downsample * self.N + 1,
            hurst=1 - (1.0 - self.hurst) / self.degree,
            sigma=1.0, random_state=self.random_state,
            n_threads=self.n_threads)

        self.fgn_.start()

        # Define the order of the Hermite polynomial
        self.hermite_coef_ = np.zeros(self.degree + 1, dtype=np.float)
        self.hermite_coef_[self.degree] = 1.
        self.initialized_ = True

    def finish(self):
        """Deinitialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            self.initialized_ = False

        self.fgn_.finish()
        self.fgn_ = None

    def draw(self):
        """Evaluate a hermite polynomial at the values of a fractional Gaussian
        Noise with the specified hurst index. Then apply the renorm-group
        transformation.
        """
        if not(hasattr(self, "initialized_") and self.initialized_):
            raise RuntimeError("The generator has not been initialized properly. "
                               "Please call `.start()` before calling `.draw()`.")

        increments = hermeval(self.fgn_.draw(), self.hermite_coef_)
        if self.n_downsample > 1:
            values_ = increments.cumsum()[self.n_downsample-1::self.n_downsample]

        else:
            values_ = increments[:-1].cumsum()

        values_ /= (self.fgn_.N - 1) ** self.fgn_.hurst
        return np.linspace(0, 1, num=self.N + 1), np.r_[0, values_]
