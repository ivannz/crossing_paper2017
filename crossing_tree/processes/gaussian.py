# -*- coding: UTF-8 -*-
"""A module with Gaussian h-sssi process generators.
"""
import numpy as np
from pyfftw import FFTW, empty_aligned

from sklearn.base import BaseEstimator as BaseGenerator
from sklearn.utils import check_random_state

class FractionalGaussianNoise(BaseGenerator):
    """A class to generate fractional Gaussian process of fixed length using
    a circulant matrix embedding method suggested by Dietrich and Newsam (1997).
    For the best performance N-1 should be a power of two.

    The circulant embedding method actually generates a pair of independent long-range
    dependent processes.
    """
    def __init__(self, N, hurst=0.5, sigma=1.0, random_state=None, n_threads=1):
        self.random_state = random_state
        self.n_threads = n_threads
        self.sigma = sigma
        self.N = N
        self.hurst = hurst

    def start(self):
        """Initialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            return

        N = self.N
        # Allocate buffers and initialize the FFTW object
        self.fft_in_ = empty_aligned(2 * N - 2, dtype=np.complex128)
        self.fft_out_ = empty_aligned(2 * N - 2, dtype=np.complex128)
## FFTW has at least two options for performance: 'FFTW_ESTIMATE' and 'FFTW_MEASURE'
##  the first one uses fast heuristics to choose an algorithm, whereas the latter actually
##  does some timining and measures of preformance of the various algorithms, and the
##  chooses the best one. Unfortunately, it takes about 2minutes for these measure, and
##  in general the speed up if insignificant.
        self.fftw_ = FFTW(self.fft_in_, self.fft_out_, threads=self.n_threads,
                          flags=('FFTW_DESTROY_INPUT', 'FFTW_ESTIMATE'),
                          direction='FFTW_FORWARD')

        # Compute the fft of the autocorrelation function.
        self.fft_acf_ = empty_aligned(2 * N - 2, dtype=np.float64)
## The autocorrelation structure for the fBM is constant provided the Hurst exponent
##  and the size sample are fixed. "Synthese de la covariance du fGn", Synthesise
##  the covariance of the fractional Gaussian noise. This autocorrelation function
##  models long range (epochal) dependence.
        R = np.arange(N, dtype=np.float128)
## The noise autocorrelation structure is directly derivable from the autocorrelation
##  of the time-continuous fBM:
##     r(s,t) = .5 * ( |s|^{2H}+|t|^{2H}-|s-t|^{2H} )
## If the noise is generated for an equally spaced. sampling of an fBM like process,
##  then the autocorrelation function must be multiplied by ∆^{2H}. Since Fourier
##  Transform is linear (even the discrete one), this routine can just generate a unit
##  variance fractional Gaussian noise.
        R = 0.5 * self.sigma * self.sigma * (
            np.abs(R - 1) ** (2.0 * self.hurst)
            + np.abs(R + 1) ** (2.0 * self.hurst)
            - 2 * np.abs(R) ** (2.0 * self.hurst))
## Generate the first row of the 2Mx2M Toeplitz matrix, where 2M = N + N-2: it should
##  be [ r_0, r_1, ..., r_{N-1}, r_{N-2}, ..., r_1 ]
        self.fft_in_[:N] = R
        self.fft_in_[:N-1:-1] = R[1:-1]
        del R

## The circulant matrix, defined by the autocorrelation structure above is necessarily
##  positive definite, which is equivalent to the FFT of any its row being non-negative.
        self.fftw_()
## Due to numerical round-off errors we truncate close to zero negative real Fourier
##  coefficients.
        self.fft_acf_[:] = np.sqrt(np.maximum(self.fft_out_.real, 0.0) / (2 * N - 2))

        # Set the random state
        self.random_state_ = check_random_state(self.random_state)
        self.queue_ = list()
        self.initialized_ = True

    def finish(self):
        """Deinitialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            self.initialized_ = False
            self.fftw_ = None
            self.fft_acf_ = None
            self.fft_in_ = None
            self.fft_out_ = None

    def draw(self):
        """Draw a single realisation of the processes trajectory from the generator.
        """
        if not(hasattr(self, "initialized_") and self.initialized_):
            raise RuntimeError("""The generator has not been initialized properly. """
                               """Please call `.start()` before calling `.draw()`.""")

        if not self.queue_:
## Basically the idea is to utilize the convolution property of the Fourier Transform
##  and multiply the transform of the autocorrelation function by the independent
##  Gaussian white noise in the frequency domain and then get back to the time domain.
##    cf. \url{ http://www.thefouriertransform.com/transform/properties.php }
## Begin with generation of a complex Gaussian white noise with unit variance and zero mean.
            self.fft_in_.real = self.random_state_.normal(size=2 * self.N - 2)
            self.fft_in_.imag = self.random_state_.normal(size=2 * self.N - 2)
## Compute the "convolution" of the circulant row (of autocorrelations) with the noise.
            self.fft_in_ *= self.fft_acf_
## "%% ATTENTION: ne pas utiliser ifft, qui utilise une normalisation differente"
## Compute this (see p.~1091 [Dietrich, Newsam; 1997]) :
##  F \times (\frac{1}{2M}\Lambda)^\frac{1}{2} \times w
            self.fftw_()
## [Dietrich, Newsam; 1997] write : "In our case the real and imaginary parts of any N
##  consecutive entries yield two independent realizations of \mathcal{N}_N(0,R) where
##  $R$ is the autocorrelation structure of an fBM."
##  Therefore take the first N complex draws to get a pair of independent realizations.
            self.queue_.append(self.fft_out_.imag[:self.N].copy())
            self.queue_.append(self.fft_out_.real[:self.N].copy())

## Generate the next sample only if needed.
        return self.queue_.pop()

class FractionalBrownianMotion(BaseGenerator):
    """A derived class to produce sample paths of a Fractional Brownian Motion with
    a specified fractional integration parameter (the Hurst exponent). For the best
    performance N should be a power of two.

    Returns a process sampled on :math:`0.0=t_0<t_1<\\ldots<t_N=1.0` with equal spacing
    given by :math:`N^{-1}`.
    """
    def __init__(self, N, hurst=0.5, random_state=None, n_threads=1):
        self.random_state = random_state
        self.n_threads = n_threads
        self.N = N
        self.hurst = hurst

    def start(self):
        """Initialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            return

        self.fgn_ = FractionalGaussianNoise(N=self.N + 1, hurst=self.hurst,
                                            sigma=self.N ** -self.hurst,
                                            random_state=self.random_state,
                                            n_threads=self.n_threads)
        self.fgn_.start()
        self.initialized_ = True

    def finish(self):
        """Deinitialize the generator.
        """
        if hasattr(self, "initialized_") and self.initialized_:
            self.initialized_ = False

        self.fgn_.finish()
        self.fgn_ = None

    def draw(self):
        """Draw a single realisation of the processes trajectory from the generator.
        """
        if not(hasattr(self, "initialized_") and self.initialized_):
            raise RuntimeError("""The generator has not been initialized properly. """
                               """Please call `.start()` before calling `.draw()`.""")

        values_ = self.fgn_.draw()[:-1].cumsum()
        return np.linspace(0, 1, num=self.N + 1), np.r_[0, values_]

