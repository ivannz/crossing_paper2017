# -*- coding: UTF-8 -*-
import numpy as np
from .fgn_pyfftw import fgn
# from .fgn_numpy import fgn

# def fbm( fgn, N, H = 0.5, time = False, **kwargs ) :
	# return fbm( N, H, time, **kwargs )

## Now derive the fractional Brownian motion for the fractional Gaussian Noise
class fbm( fgn ) :
	"""A derived class to produce sample paths of a Fractional Brownian Motion with
	a specified fractional integration parameter (the Hurst exponent). For the best
	performance N-1 should be a power of two.
	"""
	def __init__(self, N, H = 0.5, time = False, **kwargs ) :
		self.__t = np.empty( 0, np.float ) if not time else np.arange( N, dtype = np.float ) / ( N - 1 )
		fgn.__init__( self, N, H, sigma = ( 1.0 / N ) ** H, **kwargs )

	def reset( self ):
		super( fbm, self ).reset( )

	def initialize( self, numpy_random_state, **kwargs ) :
		super( fbm, self ).initialize( numpy_random_state, **kwargs )

	def __call__( self ) :
		increments = super( fbm, self ).__call__( )
		return self.__t, np.concatenate( ( [ 0 ], np.cumsum( increments[ : -1 ] ) ) )
