# -*- coding: UTF-8 -*-
import numpy as np
import pyfftw

class fgn( object ) :
	"""A class to generate fractional Gaussian process of fixed length using
	a circulant matrix method suggested by Dietrich and Newsam (1997). For the
	best performance N-1 should be a power of two.
	"""
## For better performance N-1 should be a power of two.
	def __init__( self, N, H = .5, sigma = 1.0, cache = None ) :
## The circulant embedding method actually generates a pair of independent long-range
##  dependent processes.
		self.__cache = list( ) if cache is None else cache
## Remember the sample size
		self.__N = N
## Setup a local rng
		self.__np_rand = None
## Save other parameters
		self.__H, self.__sigma = H, sigma

## Reset the internal state of the generator
	def reset( self ) :
		del self.__cache[:]

## A separate procedure for lazy initialization.
	def initialize( self, numpy_random_state, threads = 1 ) :
## Reset the internal state and initialize the random number generator
		self.reset( )
		self.__np_rand = numpy_random_state
## Aloocate the real and complex arrays
		self.__acf_ft = pyfftw.n_byte_align_empty( 2 * self.__N - 2, 32, dtype = np.float64 )
		self.__cplx_input  = pyfftw.n_byte_align_empty( 2 * self.__N - 2, 32, dtype = np.complex128 )
		self.__cplx_output = pyfftw.n_byte_align_empty( 2 * self.__N - 2, 32, dtype = np.complex128 )
## Initialize the FFT object: N - 1 is a power of two!!!
		self.__fft_object = pyfftw.FFTW( self.__cplx_input, self.__cplx_output,
## FFTW has at least two options for performance: 'FFTW_ESTIMATE' and 'FFTW_MEASURE'
##  the first one uses fast heuristics to choose an algorithm, whereas the latter actually
##  does some timining and measures of preformance of the various algorithms, and the
##  chooses the best one. Unfortunately, it takes about 2minutes for these measure, and
##  in general the speed up if insignificant.
			threads = threads, direction = 'FFTW_FORWARD', flags = ( 'FFTW_DESTROY_INPUT', 'FFTW_ESTIMATE', ) )
## The autocorrelation structure for the fBM is constant provided the Hurst exponent
##  and the size sample are fixed. "Synthese de la covariance du fGn", Synthesise
##  the covariance of the fractional Gaussian noise. This autocorrelation function
##  models long range (epochal) dependence.
		R = np.arange( self.__N, dtype = np.float64 )
## The noise autocorrelation structure is directly derivable from the autocorrelation
##  of the time-continuous fBM:
##     r(s,t) = .5 * ( |s|^{2H}+|t|^{2H}-|s-t|^{2H} )
## If the noise is generated for an equally spaced. sampling of an fBM like process,
##  then the autocorrelation function must be multiplied by ∆^{2H}. Since Fourier
##  Transform is linear (even the discrete one), this routine can just generate a unit
##  variance fractional Gaussian noise.
		R = self.__sigma * self.__sigma * .5 * (
			  np.abs( R - 1 ) ** ( 2.0 * self.__H )
			+ np.abs( R + 1 ) ** ( 2.0 * self.__H )
			- 2 * np.abs( R ) ** ( 2.0 * self.__H ) )
## Generate the first row of the 2Mx2M Toeplitz matrix, where 2M = N + N-2: it should
##  be [ r_0, ..., r_{N-1}, r_{N-2}, ..., r_1 ]
		self.__cplx_input[:] = np.append( R, R[::-1][1:-1] ) + 1j * 0
		del R
## The circulant matrix, defined by the autocorrelation structure above is necessarily
##  positive definite, which is equivalent to the FFT of any its row being non-negative.
		self.__fft_object( )
## Due to numerical round-off errors we truncate close to zero negative real Fourier
##  coefficients.
		self.__acf_ft[:] = np.sqrt( np.maximum( np.real( self.__cplx_output ), 0.0 ) / ( 2 * self.__N - 2 ) )

## fGn generator via circulant embedding method
	def __gen( self ) :
## Basically the idea is to utilize the convolution property of the Fourier Transform
##  and multiply the transform of the autocorrelation function by the independent
##  Gaussian white noise in the frequency domain and then get back to the time domain.
##    cf. \url{ http://www.thefouriertransform.com/transform/properties.php }
## Begin with generation of the Gaussian white noise with unit variance and zero mean.
		self.__cplx_input[:] = self.__np_rand.randn( 2 * self.__N - 2 ) + 1j * self.__np_rand.randn( 2 * self.__N - 2 )
## Compute the "convolution" of the circulant row (of autocorrelations) with the noise.
		self.__cplx_input *= self.__acf_ft
## "%% ATTENTION: ne pas utiliser ifft, qui utilise une normalisation differente"
## Compute this (see p.~1091 [Dietrich, Newsam; 1997]) :
##  F \times (\frac{1}{2M}\Lambda)^\frac{1}{2} \times w
		self.__fft_object( )
## [Dietrich, Newsam; 1997] write : "In our case the real and imaginary parts of any N
##  consecutive entries yield two independent realizations of \mathcal{N}_N(0,R) where
##  $R$ is the autocorrelation structure of an fBM."
##  Therefore take the first N complex draws to get a pair of independent realizations.
## Return views into the real and the imaginary parts of the generated array of complex
##  numbers -- this is cool and optimal.
		return ( np.real( self.__cplx_output[ :self.__N ] ), np.imag( self.__cplx_output[ :self.__N ] ) )

	def __del__( self ) :
		self.reset( )

## A visbile function, to generate the sample
	def __call__( self ) :
## Generate the next sample only if needed.
		if not self.__cache :
			self.__cache.extend( self.__gen( ) )
## Return a pregenerated sample
		return self.__cache.pop( )
