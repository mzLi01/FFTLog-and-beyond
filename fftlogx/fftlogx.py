"""
python module for calculating integrals with 1 Bessel functions.
This module contains:
-- FFTLog method for integrals with 1 spherical Bessel function;
-- integrals with 1 (cylindrical) Bessel function, i.e. Hankel transform;
-- window function (optional) for smoothing Fourier coefficients

by Xiao Fang
Apr 10, 2019
"""

import numpy as np
from .lib_wrapper import *

class fftlog(object):

	def __init__(self, x, fx, nu=1.1, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
		"""
		fftlog class for integrals with 1 spherical Bessel function.
		Parameters:
		-----------
		x: array
			x values, should be logarithmically spaced
		fx: array
			f(x) values
		nu: float
			the bias index, should be between -ell to 2
		N_extrap_low, N_extrap_high: int
			Number of points to extrapolate lower than x[0] and higher than x[-1]
		c_window_width: float
			The fraction of c_m elements (Fourier coefficients of "biased" input function f(x): f_b = f(x) / x^\nu) that are smoothed,
			e.g. c_window_width=0.25 means smoothing the last 1/4 of c_m elements using "c_window".
		N_pad: int
			Number of zero-padding on both sides of the input function f(x)
		"""
		self.x_origin = x # x is logarithmically spaced
		dlnx = np.log(x[1:]/x[:-1])
		if np.allclose(dlnx, dlnx[0]):
			self.dlnx = dlnx[0]
		else:
			raise ValueError('x should be logarithmically spaced')
		self.fx_origin= fx # f(x) array
		self.nu = nu
		self.N_extrap_low = N_extrap_low
		self.N_extrap_high = N_extrap_high
		self.c_window_width = c_window_width

		# extrapolate x and f(x) linearly in log(x), and log(f(x))
		self.x = log_extrap(x, N_extrap_low, N_extrap_high)
		self.fx = log_extrap(fx, N_extrap_low, N_extrap_high)
		self.N = self.x.size

		# zero-padding
		self.N_pad = N_pad
		if(N_pad):
			pad = np.zeros(N_pad)
			self.x = log_extrap(self.x, N_pad, N_pad)
			self.fx = np.hstack((pad, self.fx, pad))
			self.N += 2*N_pad
			self.N_extrap_high += N_pad
			self.N_extrap_low += N_pad

		if(self.N%2==1): # Make sure the array sizes are even
			self.x= self.x[:-1]
			self.fx=self.fx[:-1]
			self.N -= 1
			if(N_pad):
				self.N_extrap_high -=1

	def _check_nu(self, ell1, ell2, bessel_type: BesselIntType):
		if bessel_type in (BesselIntType.J_ELL, BesselIntType.J_ELL_DERIV, BesselIntType.J_ELL_DERIV_2, BesselIntType.J_ELL_SQUARED):
			nu_low = -np.max(ell1)
			nu_low_str = '-ell1'
			nu_high = 2
			nu_high_str = '2'
		elif bessel_type == BesselIntType.J_ELL1_J_ELL2_EQ:
			nu_low = np.max(-ell1-ell2)
			nu_low_str = '-(ell1+ell2)'
			nu_high = 2
			nu_high_str = '2'
		elif bessel_type == BesselIntType.J_ELL1_J_ELL2_NEQ:
			nu_low = np.max(-ell1-ell2)
			nu_low_str = '-(ell1+ell2)'
			nu_high = 3
			nu_high_str = '3'
		else:
			nu_low = -2*np.max(ell1)
			nu_low_str = '-2 ell'
			nu_high = 3
			nu_high_str = '3'
		if self.nu < nu_low or self.nu > nu_high:
			raise ValueError(
				f'nu is required to be between {nu_low_str} to {nu_high_str}')

	def _check_y_ratio(self, y_ratio, ells1, ells2):
		if y_ratio > 1:
			return 1/y_ratio, ells2, ells1
		elif y_ratio < 1:
			return y_ratio, ells1, ells2
		else:
			raise ValueError('use _eq methods for y_ratio=1 cases!')

	def _fftlog(self, ell1: int, bessel_type: BesselIntType, ell2: int = -1, y_ratio: float = 1):
		"""
		cfftlog wrapper
		"""
		y, Fy = self._fftlog_ells(
			np.array([ell1]), bessel_type, ells2=np.array([ell2]), y_ratio=y_ratio)
		return y[0], Fy[0]

	def _fftlog_ells(self, ells1: np.ndarray, bessel_type: BesselIntType, ells2=None, y_ratio: float = 1):
		"""
		cfftlog_ells wrapper
		"""
		if ells2 is None:
			ells2 = -1*np.ones_like(ells1)
		self._check_nu(ells1, ells2, bessel_type)
		Nell = ells1.shape[0]
		y = np.zeros((Nell,self.N))
		Fy= np.zeros((Nell,self.N))
		ypp = (y.__array_interface__['data'][0] 
      			+ np.arange(Nell)*y.strides[0]).astype(np.uintp) 
		Fypp = (Fy.__array_interface__['data'][0] 
				+ np.arange(Nell)*Fy.strides[0]).astype(np.uintp)

		cfftlog_ells_wrapper(np.ascontiguousarray(self.x, dtype=np.float64),
						np.ascontiguousarray(self.fx, dtype=np.float64),
						clong(self.N),
						cdouble(y_ratio),
						np.ascontiguousarray(ells1, dtype=np.float64),
						np.ascontiguousarray(ells2, dtype=np.float64),
						clong(Nell),
						ypp,
						Fypp,
						cdouble(self.nu),
						cdouble(self.c_window_width),
						bessel_type,
						clong(0) # Here N_pad is done in python, not in C
						)

		return y[:,self.N_extrap_high:self.N-self.N_extrap_low], Fy[:,self.N_extrap_high:self.N-self.N_extrap_low]

	def _fftlog_modified_ells(self, ell_array, derivative, j_squared=0):
		"""
		cfftlog_ells wrapper
		"""
		if(j_squared!=0):
			raise ValueError('j_squared Not supported for modified fftlog!')

		Nell = ell_array.shape[0]
		y = np.zeros((Nell,self.N))
		Fy= np.zeros((Nell,self.N))
		ypp = (y.__array_interface__['data'][0] 
      			+ np.arange(Nell)*y.strides[0]).astype(np.uintp) 
		Fypp = (Fy.__array_interface__['data'][0] 
				+ np.arange(Nell)*Fy.strides[0]).astype(np.uintp)

		cfftlog_modified_ells_wrapper(np.ascontiguousarray(self.x, dtype=np.float64),
						np.ascontiguousarray(self.fx, dtype=np.float64),
						clong(self.N),
						np.ascontiguousarray(ell_array, dtype=np.float64),
						clong(Nell),
						ypp,
						Fypp,
						cdouble(self.nu),
						cdouble(self.c_window_width),
						cint(derivative),
						cint(j_squared),
						clong(0) # Here N_pad is done in python, not in C
						)

		return y[:,self.N_extrap_high:self.N-self.N_extrap_low], Fy[:,self.N_extrap_high:self.N-self.N_extrap_low]


	def fftlog(self, ell:int, n:int=0):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j^{(n)}_\ell(xy),
		where j^{(n)}_\ell is the n-th derivative of spherical Bessel func of order ell.
		Value of n should be 0, 1, or 2.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		type_map = {0: BesselIntType.J_ELL, 1: BesselIntType.J_ELL_DERIV, 2: BesselIntType.J_ELL_DERIV_2}
		return self._fftlog(ell, type_map[n])

	def fftlog_jsqr(self, ell:int):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * |j_\ell(xy)|^2,
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog(ell, BesselIntType.J_ELL_SQUARED)

	def fftlog_ells(self, ell_array:np.ndarray, n:int=0):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j^{(n)}_\ell(xy),
		where j^{(n)}_\ell is the n-th derivative of spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		type_map = {0: BesselIntType.J_ELL, 1: BesselIntType.J_ELL_DERIV, 2: BesselIntType.J_ELL_DERIV_2}
		return self._fftlog_ells(ell_array, type_map[n])

	def fftlog_jsqr_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_ells(ell_array, BesselIntType.J_ELL_SQUARED)

	def jl1_jl2_eq(self, ells1, ells2):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j_{\ell_1}(xy) * j_{\ell_2}(xy)
		"""
		return self._fftlog_ells(ells1, BesselIntType.J_ELL1_J_ELL2_EQ, ells2=ells2)
	
	def jl1_jl2_neq(self, ells1, ells2, y_ratio):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j_{\ell_1}(xy) * j_{\ell_2}(beta*xy),
		where y_ratio is beta.
		"""
		y_ratio, ells1, ells2 = self._check_y_ratio(y_ratio, ells1, ells2)
		return self._fftlog_ells(ells1, BesselIntType.J_ELL1_J_ELL2_NEQ, ells2=ells2, y_ratio=y_ratio)

	def fftlog_modified_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x)/(xy)^2 * j_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_modified_ells(ell_array, 0)

	def fftlog_dj_modified_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x)/(xy)^2 * j'_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_modified_ells(ell_array, 1)

	def fftlog_ddj_modified_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x)/(xy)^2 * j''_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_modified_ells(ell_array, 2)


class hankel(object):
	def __init__(self, x, fx, nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
		self.nu = nu
		self.myfftlog = fftlog(x, np.sqrt(x)*fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)

	def _check_nu(self, n):
		if self.nu < np.max(0.5-n) or self.nu > 2:
			raise ValueError('nu is required to be between (0.5-n) and 2.')

	def hankel(self, n):
		self._check_nu(n)
		y, Fy = self.myfftlog.fftlog(n-0.5)
		Fy *= np.sqrt(2*y/np.pi)
		return y, Fy

	def hankel_narray(self, n_array):
		self._check_nu(n_array)
		y, Fy = self.myfftlog.fftlog_ells(n_array-0.5)
		Fy *= np.sqrt(2*y/np.pi)
		return y, Fy


### Utility functions ####################

def log_extrap(x, N_extrap_low, N_extrap_high):

	low_x = []
	high_x = []
	if(N_extrap_low):
		dlnx_low = np.log(x[1]/x[0])
		low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0) )
	if(N_extrap_high):
		dlnx_high= np.log(x[-1]/x[-2])
		high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high+1) )
	x_extrap = np.hstack((low_x, x, high_x))
	return x_extrap
