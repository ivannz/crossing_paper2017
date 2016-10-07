# -*- coding: UTF-8 -*-
import numpy as np

from numpy.polynomial.hermite_e import hermeval

from .gaussian import FractionalGaussianNoise

class HermiteProcess(FractionalGaussianNoise):
    """A derived class to produce sample paths of a Hermite process of order `degree`
    with a specified Hurst exponent (fractional integration parameter). For the best
    performance `N-1` should be a power of two. The hurst exponent for this process
    is :math:`H = 1 + degree * (H_{\\text{fgn}} - 1)`.

    Details
    -------
    When downsampling parameter (`n_downsample`) tends to infinity
    the process, converges in distribution to the Rosenblatt process or in general to
    a Hermite process. This stems from the `non-central limit theorem`:
    .. math :
        Z^k(t) = \\frac{1}{n^\\alpha}\\sum_{j=1}^{\\lfloor kt\\rfloor} H(\\xi_j) \,,
    converges to :math:`Z_\\frac{\\alpha}{2}(t)` -- a hermite process. Thus increasing
    `n_downsample` gives better approximation.

    In theory it should tend to infinity. This is a serious drawback. c.f. [Abry, Pipiras; 2005]
    """
    def __init__(self, N, degree=2, n_downsample=16, hurst=0.5,
                 random_state=None, n_threads=1):
        super(HermiteProcess, self).__init__(N=n_downsample * (N - 1) + 1,
                                             hurst=(hurst + degree - 1.0) / degree,
                                             sigma=1.0,
                                             random_state=random_state,
                                             n_threads=n_threads)
        self.degree = degree
        self.n_downsample = n_downsample
        
    def start(self):
        super(HermiteProcess, self).start()

        # Define the order of the Hermite polynomial
        self.hermite_coef_ = np.zeros(self.degree + 1, dtype=np.float)
        self.hermite_coef_[self.degree] = 1.

    def draw(self):
        """Evaluate a hermite polynomial at the values of a fractional Gaussian Noise
        with the specified hurst index. Then apply the renorm-group transformation
        omitting the renormalisation factor :math: `n^{-H}`.
        """
        increments = hermeval(super(FractionalBrownianMotion, self).draw(),
                              self.hermite_coef_)
        values_ = increments.cumsum()[self.n_downsample-1::self.n_downsample]
        return np.linspace(0, 1, num=self.N+1), values_
