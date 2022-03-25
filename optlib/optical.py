import subprocess
import numpy as np
import math
import matplotlib.pyplot as plt
import nlopt
import copy
import pandas as pd
import os
from scipy import special, interpolate, sparse, stats, optimize
import time
from tqdm import tqdm

hc = 12.3981756608  # planck constant times velocity of light keV Angstr
r0 = 2.8179403227e-15
h2ev = 27.21184  # Hartree, converts Hartree to eV
a0 = 0.529177  # Bohr Radius in Angstroem
machine_eps = np.finfo('float64').eps
wpc = 4*math.pi * a0**3
N_Avogadro = 6.02217e23
c = 137.036

def wavelength2energy(wavelength):
	# wavelength in Angstroem
    return hc / wavelength  * 1e3


def linspace(start, stop, step=1.):
	num = int((stop - start) / step + 1)
	return np.linspace(start, stop, num)


def check_list_type(lst, type_to_check):
	if lst and isinstance(lst, list):
		return all(isinstance(elem, type_to_check) for elem in lst)
	elif lst:
		return isinstance(lst, type_to_check)
	else:
		return True


def is_list_of_int_float(lst):
	if lst and isinstance(lst, list):
		return all(isinstance(elem, int) or isinstance(elem, float) for elem in lst)
	elif lst:
		return (isinstance(lst, int) or isinstance(lst, float))
	else:
		return True

def conv(x1, x2, de, mode='right'):
	n = x1.size
	a = np.convolve(x1, x2)
	if mode == 'right':
		return a[0:n] * de
	elif mode == 'left':
		return a[a.size-n:a.size] * de
	else:
		return a * de

def gauss(x, a1, b1, c1):
    return a1*np.exp(-((x-b1)/c1)**2)

class Error(Exception):
	"""Base class for exceptions in this module."""
	pass


class InputError(Error):
	"""Exception raised for errors in the input.

	Attributes:
		message -- explanation of the error
	"""

	def __init__(self, message):
		self.message = message


class Composition:
	def __init__(self, elements, indices):
		if not check_list_type(elements, str):
			raise InputError(
				"The array of elements passed must be of the type string!")
		if not is_list_of_int_float(indices):
			raise InputError(
				"The array of indices passed must be of the type int or float!")
		if isinstance(elements, list) and isinstance(indices, list) and len(elements) != len(indices):
			raise InputError(
				"The number of elements and indices must be the same!")
		self.elements = elements
		self.indices = indices


class Oscillators:
	def __init__(self, model, A, gamma, omega, alpha = 1.0, eps_b = 1.0):
		if not is_list_of_int_float(omega):
			raise InputError(
				"The array of omega passed must be of the type int or float!")
		if not is_list_of_int_float(gamma):
			raise InputError(
				"The array of gamma passed must be of the type int or float!")
		if not is_list_of_int_float(A):
			raise InputError(
				"The array of A passed must be of the type int or float!")
		if omega and A and gamma:
			if len(omega) != len(A) != len(gamma):
				raise InputError(
					"The number of oscillator parameters must be the same!")
		self.model = model;
		self.A = np.array(A);
		self.gamma = np.array(gamma);
		self.omega = np.array(omega);
		self.alpha = alpha;
		self.eps_b = eps_b;


class ExperimentConfiguration:
	def __init__(self, theta0 = 0, theta = 0, phi = 0):
		self.theta0 = theta0
		self.theta = theta
		self.phi = phi


class Material:
	'''Here will be some description of the class Material'''

	def __init__(self, name, oscillators, composition, eloss, q, xraypath):
		if not isinstance(oscillators, Oscillators):
			raise InputError("The oscillators must be of the type Oscillators")
		if not isinstance(composition, Composition):
			raise InputError("The composition must be of the type Composition")
		self.name = name
		self.oscillators = oscillators;
		self.composition = composition
		self.eloss = np.array(eloss)
		self.eloss[eloss == 0] = 1e-5
		self.xraypath = xraypath
		self.Eg = 0
		self.Ef = 0
		self.U = 0
		self.width_of_the_valence_band = None
		self.atomic_density = None
		self.static_refractive_index = None
		self.electron_density = None
		self.Z = None
		self.ELF = None
		self.ELF_extended_to_Henke = None
		self.ELF_Henke = None
		self.eloss_Henke = None
		self.surfaceELF = None
		# self.epsilon = None
		self.DIIMFP = None
		self.DIIMFP_E = None
		self.DECS = None
		self.DECS_mu = None
		self.E0 = None
		self.IMFP = None
		self.IMFP_E = None
		self.EMFP = None
		self.sigma_el = None
		self.q_dependency = None
		self._q = q
		self.use_KK_constraint = False
		self.use_henke_for_ne = False
		self.electron_density_Henke = 0
		self.use_kk_relation = False

	@property
	def q(self):
		if isinstance(self._q, np.ndarray):
			return self._q
		elif isinstance(self._q,list):
			return np.array(self._q)
		else:
			return self._q
	
	@q.setter
	def q(self, q):
		try: 
			self.size_q = q.shape[1]
		except IndexError:
			self.size_q = q.shape[0]
		except:
			self.size_q = 1
		self._q = q

	@property
	def epsilon(self):
		self.calculateDielectricFunction()
		return self._epsilon

	def kramers_kronig(self, epsilon_imag):
		eps_real = np.zeros_like(self.eloss)
		for i in range(self.eloss.size):
			omega = self.eloss[i]
			ind = np.all([self.eloss != omega, self.eloss > self.Eg], axis=0)
			if len(epsilon_imag.shape) > 1:
				kk_sum = np.trapz(self.eloss[ind] * epsilon_imag[ind][0] / (self.eloss[ind] ** 2 - omega ** 2), self.eloss[ind])
			else:
				kk_sum = np.trapz(self.eloss[ind] * epsilon_imag[ind] / (self.eloss[ind] ** 2 - omega ** 2), self.eloss[ind])
			eps_real[i] = 2 * kk_sum / math.pi + 1
		return eps_real

	def calculateDielectricFunction(self):
		if self.oscillators.model == 'Drude':
			self._epsilon = self.calculateDrudeDielectricFunction()
		elif self.oscillators.model == 'DrudeLindhard':
			self._epsilon = self.calculateDLDielectricFunction()
		elif self.oscillators.model == 'Mermin':
			self._epsilon = self.calculateMerminDielectricFunction()
		elif self.oscillators.model == 'MerminLL':
			self._epsilon = self.calculateMerminLLDielectricFunction()
		else:
			raise InputError("Invalid model name. The valid model names are: Drude, DrudeLindhard, Mermin and MerminLL")

	def calculateDrudeDielectricFunction(self):
		self.convert2au()

		for i in range(len(self.oscillators.A)):
			epsDrude_real, epsDrude_imag = self.calculateDrudeOscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i], self.oscillators.alpha)
			if i == 0:
				eps_real = self.oscillators.eps_b * np.ones_like(epsDrude_real)
				eps_imag = np.zeros_like(epsDrude_imag)
				epsilon = np.zeros_like(eps_real, dtype=complex)
			eps_real -= self.oscillators.A[i] * epsDrude_real
			eps_imag += self.oscillators.A[i] * epsDrude_imag

		if self.Eg > 0:
			eps_imag[self.eloss <= self.Eg] = 1e-5
		if self.use_kk_relation:
			eps_real = self.kramers_kronig(eps_imag)

		epsilon.real = eps_real
		epsilon.imag = eps_imag

		self.convert2ru()
		return epsilon

	def calculateDrudeOscillator(self, omega0, gamma, alpha):
		# if not self.q_dependency is None:
		# 	w_at_q = omega0 + 0.5 * alpha * self.q**0.5
			# w_at_q = omega0 + (self.q_dependency(self.q / a0)/h2ev - self.q_dependency(0)/h2ev)
		# else:
		w_at_q = omega0 + 0.5 * alpha * self.q**2
		if self.size_q == 1:
			omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())
		else:
			omega = np.expand_dims(self.eloss, axis=tuple(range(1,self.q.ndim)))

		mm = omega**2 - w_at_q**2
		divisor = mm**2 + omega**2 * gamma**2

		eps_real = mm / divisor
		eps_imag = omega*gamma / divisor

		return eps_real, eps_imag

	def calculateDLDielectricFunction(self):
		self.convert2au()
		epsilon = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		sum_oneover_eps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		oneover_eps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

		for i in range(len(self.oscillators.A)):
			oneover_eps = self.calculateDLOscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i], self.oscillators.alpha)
			sum_oneover_eps += self.oscillators.A[i] * (oneover_eps - complex(1))

		sum_oneover_eps += complex(1)
		epsilon = complex(1) / sum_oneover_eps

		if self.use_kk_relation:
			eps_imag = epsilon.imag
			eps_real = self.kramers_kronig(eps_imag)
			epsilon.real = eps_real
			epsilon.imag = eps_imag

		self.convert2ru()
		return epsilon

	def calculateDLOscillator(self, omega0, gamma, alpha):
		if not self.q_dependency is None:
			w_at_q = omega0 - self.q_dependency(0)/h2ev + self.q_dependency(self.q / a0)/h2ev
		else:
			w_at_q = omega0 + 0.5 * alpha * self.q**2

		omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())

		mm = omega**2 - w_at_q**2
		divisor = mm**2 + omega**2 * gamma**2

		one_over_eps_imag = -omega0**2 * omega * gamma / divisor
		if self.Eg > 0:
			one_over_eps_imag[self.eloss <= self.Eg] = 1e-5
		if self.use_kk_relation:
			one_over_eps_real = self.kramers_kronig(one_over_eps_imag)
		else:
			one_over_eps_real = 1.0 + omega0**2 * mm / divisor

		one_over_eps = np.squeeze(np.apply_along_axis(lambda args: [complex(
			*args)], 0, np.array([one_over_eps_real, one_over_eps_imag])))

		return one_over_eps

	def calculateMerminDielectricFunction(self):
		if self.size_q == 1 and self.q == 0:
			self.q = 0.01
		self.convert2au()
		epsilon = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		oneovereps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

		for i in range(len(self.oscillators.A)):
			if all(np.abs((self.eloss - self.oscillators.omega[i]) / self.oscillators.gamma[i]) < 100000):
				epsMermin = self.calculateMerminOscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i])
			else:
				epsMermin = complex(1)
			oneovereps += self.oscillators.A[i] * (complex(1) / epsMermin)
		oneovereps += complex(1) - complex(np.sum(self.oscillators.A))
		
		if self.Eg > 0:
			oneovereps.imag[self.eloss <= self.Eg] = 1e-5

		epsilon = complex(1) / oneovereps
		if self.use_kk_relation:
			eps_imag = epsilon.imag
			eps_real = self.kramers_kronig(eps_imag)
			epsilon.real = eps_real
			epsilon.imag = eps_imag

		self.convert2ru()
		return epsilon

	def calculateLinhardOscillator(self, omega, gamma, omega0):
		n_dens = omega0**2 / (4*math.pi)
		E_f = 0.5 * (3 * math.pi**2 * n_dens)**(2.0 / 3.0)
		v_f = (2*E_f)**0.5
		
		z = self.q / (2 * v_f);  
		chi = np.sqrt(1.0 / (math.pi * v_f))
		
		z1_1 = omega / (self.q * v_f)
		z1_1[np.isnan(z1_1)] = machine_eps
		
		gq = np.zeros_like(self.q)
		gq = gamma / (self.q * v_f)
		vos_g_array = np.vectorize(self.vos_g)
		reD1, imD1 = vos_g_array(z1_1 + z, gq)
		reD2, imD2 = vos_g_array(z1_1 - z, gq)
		
		red1_d2 = reD1 - reD2
		imd1_d2 = imD1 - imD2
		
		chizzz = chi**2 / (z**3 * 4)
		epsreal = 1 + red1_d2 * chizzz
		epsimag = imd1_d2 * chizzz
		complex_array = np.vectorize(complex)
		return complex_array(epsreal, epsimag)

	def vos_g(self, z, img_z):
		zplus1 = z + 1
		zminus1 = z - 1
		
		if img_z != 0:
			imgZ2 = img_z**2
			dummy1 = math.log( np.sqrt((zplus1 * zplus1 + imgZ2) / (zminus1 * zminus1 + imgZ2)) )
			dummy2 = math.atan2(img_z, zplus1) - math.atan2(img_z, zminus1)

			reim1 = 1 - (z**2 - imgZ2)

			outreal_1 = z + 0.5 * reim1 * dummy1
			outreal = outreal_1 + z *img_z * dummy2

			outimag_1 = img_z + 0.5 * reim1 * dummy2
			outimag = outimag_1 - z * img_z * dummy1
		else:
			dummy1 = math.log( abs(zplus1) / abs(zminus1) )
			dummy2 = math.atan2(0, zplus1) - math.atan2(0, zminus1)

			reim1 = 1 - z**2

			outreal_1 = z + 0.5 * reim1 * dummy1
			outreal = outreal_1

			outimag_1 = 0.5 * reim1 * dummy2
			outimag = outimag_1
		return outreal, outimag

	def calculateMerminOscillator(self, omega0, gamma):
		omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())
		gammma_over_omega = gamma / omega
		complex_array = np.vectorize(complex)
		z1 = complex_array(1, gammma_over_omega)
		z2 = self.calculateLinhardOscillator(omega, gamma, omega0) - complex(1)
		z3 = self.calculateLinhardOscillator(np.zeros_like(omega), 0, omega0) - complex(1)
		top = z1 * z2
		bottom = complex(1) + complex_array(0, gammma_over_omega) * z2 / z3
		return complex(1) + top / bottom

	def calculateMerminLLDielectricFunction(self):
		if self.U == 0:
			raise InputError("Please specify the value of U")
		if self.size_q == 1 and self.q == 0:
			self.q = 0.01
		self.convert2au()
		epsilon = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		oneovereps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

		for i in range(len(self.oscillators.A)):
			epsMLL = self.calculateMerminLLOscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i])
			oneovereps += self.oscillators.A[i] * (complex(1) / epsMLL - complex(1))
		oneovereps += complex(1)
		epsilon = complex(1) / oneovereps
		self.convert2ru()
		return epsilon

	def convert_to_MLL(self):
		currentU = 0.0
		newsumA = 0.0
		stopthis = 0
		UstepSize = 1.0
		counter = 0
		while (newsumA < 1) and (currentU < 1000):
			oldsumA = newsumA
			newsumA = 0.0
			currentU = currentU + UstepSize
			print("currentU", currentU, "UstepSize", UstepSize)
			#print("currentU", currentU)
			for i in range(len(self.oscillators.A)):
				oldA = self.oscillators.A[i]
				if oldA > 0.0:
					old_omega = self.oscillators.omega[i]
					if old_omega <= currentU:
						stopthis = 1
						break
					new_omega = math.sqrt(old_omega**2 - currentU**2)
					NewA = oldA * (old_omega / new_omega)**2
					newsumA = newsumA + NewA
					if newsumA > 1:
						stopthis = 1
						break

			if stopthis:
				currentU = currentU - UstepSize  # go back to previous U value
				newsumA = oldsumA  # make step size 10 times smaller
				UstepSize = UstepSize / 10.0
				stopthis = 0
				print("newsumA", newsumA)
				counter = counter + 1
				if counter > 100:
					break
				if  newsumA > 0.99:
					break

		for i in range(len(self.oscillators.A)):  # now calculate new values for optimum U
			oldA = self.oscillators.A[i]
			if oldA > 0.0:
				old_omega = self.oscillators.omega[i]
				if old_omega < currentU:
					new_omega = 0.001
				else:
					new_omega = math.sqrt(old_omega**2 - currentU**2)
				NewA = oldA * (old_omega / new_omega)**2
				self.oscillators.A[i] = NewA
				self.oscillators.omega[i] = new_omega
		self.U = currentU

	def calculateMerminLLOscillator(self, omega0, gamma):
		omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())
		gammma_over_omega = gamma / omega
		complex_array = np.vectorize(complex)
		z1 = complex_array(1, gammma_over_omega)
		z2 = self.eps_LLX(omega, gamma, omega0) - complex(1)
		z3 = self.eps_LLX(np.zeros_like(omega), 0, omega0) - complex(1)
		top = z1 * z2
		bottom = complex(1) + complex_array(0, gammma_over_omega) * z2 / z3
		return complex(1) + top / bottom

	def eps_LLX(self, omega, gamma, omega0):
		complex_array = np.vectorize(complex)
		omega_minus_square = complex_array(omega**2 - self.U**2 - gamma**2, 2.0 * omega * gamma)
		r = np.abs(omega_minus_square)
		theta = np.arctan2(omega_minus_square.imag, omega_minus_square.real)
		omega_minus = complex_array(np.sqrt(r) * np.cos(theta / 2.0), np.sqrt(r) * np.sin(theta / 2.0))
		epsilon = np.zeros_like(omega_minus)
		ind_ge = omega_minus.real >= 0
		ind_lt = omega_minus.real < 0
		ge = any(ind_ge.flatten())
		lt = any(ind_lt.flatten())
		if ge:
			epsilon[ind_ge] = self.calculateLinhardOscillator(omega_minus.real, omega_minus.imag, omega0)[ind_ge]      
		if lt:
			n_dens = omega0**2 / (4.0 * math.pi)
			E_f = 0.5 * (3.0 * math.pi**2 * n_dens)**(2.0 / 3.0)
			v_f = (2 * E_f)**0.5
			DeltaSquare = -omega_minus_square / E_f**2
			r = abs(DeltaSquare)
			# theta = atan2(DeltaSquare.imag, DeltaSquare.real)
			theta = np.arctan2(DeltaSquare.imag, DeltaSquare.real)
			Delta = complex_array(np.sqrt(r) * np.cos(theta / 2.0), np.sqrt(r) * np.sin(theta / 2.0))
			QQ = self.q / v_f
			z1 = 2.0 * QQ + QQ**2
			res1 = z1 / Delta
			res1 = self.c_arctan(res1)
			z2 = 2.0 * QQ + QQ**2
			res2 = z2 / Delta
			res2 = self.c_arctan(res2)
			res1 = res1 + res2
			res2 = res1 * Delta
			
			z1 = complex_array( DeltaSquare.real + (2 * QQ + QQ**2)**2 , DeltaSquare.imag)
			z2 = complex_array( DeltaSquare.real + (2 * QQ - QQ**2)**2 , DeltaSquare.imag)
			z1 = z1 / z2
			z1 = np.log(z1)
			z2 = DeltaSquare * z1
			
			p1 = res2.imag / (2 * QQ**3)
			p2 = z2.imag / (8 * QQ**5)
			p3 = z1.imag / (2 * QQ**3)
			p4 = z1.imag / (8 * QQ)    
			eps_imag = 2 / (math.pi * v_f) * (-p1 + p2 + p3 - p4)
			
			t1 = res2.real / (2 * QQ**3)
			t2 = z2.real / (8 * QQ**5)
			t3 = z1.real / (2 * QQ**3)
			t4 = z1.real / (8 * QQ)
			t5 = 1 / QQ**2 - t1
			eps_real = 1 + 2 / (math.pi * v_f) * (t5 + t2 + t3 - t4)
			
			epsilon[ind_lt] = complex_array(eps_real, eps_imag)[ind_lt]
		return epsilon

	def c_arctan(self, z):
		complex_array = np.vectorize(complex)
		reres = np.zeros_like(z)
		x = z.real
		y = z.imag
		imres = -1.0 / 4.0 * np.log( (1 - x**2 - y**2)**2 + 4 * x**2) \
				+ 1.0 / 2.0 * np.log( (1 + y)**2 + x**2 )
		reres[x != 0] = math.pi / 4.0 - 0.5 * np.arctan( (1 - x**2 - y**2) / (2.0 * x) )
		reres[np.logical_and(x > 0, x < 0)] = math.pi / 2.0
		return complex_array(reres.real, imres.imag)

	def convert2au(self):
		if self.oscillators.model == 'Drude':
			self.oscillators.A = self.oscillators.A/h2ev/h2ev
		self.oscillators.gamma = self.oscillators.gamma/h2ev
		self.oscillators.omega = self.oscillators.omega/h2ev
		self.Ef = self.Ef/h2ev
		self.eloss = self.eloss/h2ev
		self.q = self.q*a0
		if (self.Eg):
			self.Eg = self.Eg/h2ev
		if (self.U):
			self.U = self.U/h2ev
		if (self.width_of_the_valence_band):
			self.width_of_the_valence_band = self.width_of_the_valence_band/h2ev

	def convert2ru(self):
		if self.oscillators.model == 'Drude':
			self.oscillators.A = self.oscillators.A*h2ev*h2ev
		self.oscillators.gamma = self.oscillators.gamma*h2ev
		self.oscillators.omega = self.oscillators.omega*h2ev
		self.Ef = self.Ef*h2ev
		self.eloss = self.eloss*h2ev
		self.q = self.q/a0
		if (self.Eg):
			self.Eg = self.Eg*h2ev
		if (self.U):
			self.U = self.U*h2ev
		if (self.width_of_the_valence_band):
			self.width_of_the_valence_band = self.width_of_the_valence_band*h2ev

	def evaluateFsum(self):
		old_q = self.q
		self.q = 0
		self.extendToHenke()
		fsum = 1 / (2 * math.pi**2 * (self.atomic_density * a0**3)) * np.trapz(self.eloss_extended_to_Henke/h2ev * self.ELF_extended_to_Henke, self.eloss_extended_to_Henke/h2ev)
		self.q = old_q
		return fsum

	def evaluateKKsum(self):
		old_q = self.q
		self.q = 0
		if (self.oscillators.model == 'MerminLL'):
			self.q = 0.01
		self.extendToHenke()
		div = self.ELF_extended_to_Henke / self.eloss_extended_to_Henke
		div[((div < 0) | (np.isnan(div)))] = 1e-5
		kksum = 2 / math.pi * np.trapz(div, self.eloss_extended_to_Henke)
		if self.Eg != 0:
			kksum += 1 / self.static_refractive_index**2
		self.q = old_q
		return kksum

	def extendToHenke(self):
		self.calculateELF()
		if self.eloss_Henke is None and self.ELF_Henke is None:
			self.eloss_Henke, self.ELF_Henke = self.mopt()
		ind = self.eloss < 100
		self.eloss_extended_to_Henke = np.concatenate((self.eloss[ind], self.eloss_Henke))
		self.ELF_extended_to_Henke = np.concatenate((self.ELF[ind], self.ELF_Henke))

	def mopt(self):
		if self.atomic_density is None:
			raise InputError("Please specify the value of the atomic density atomic_density")
		numberOfElements = len(self.composition.elements)
		energy = linspace(100,30000)
		f1sum = np.zeros_like(energy)
		f2sum = np.zeros_like(energy)

		for i in range(numberOfElements):
			dataHenke = self.readhenke(self.xraypath + self.composition.elements[i])
			f1 = np.interp(energy, dataHenke[:, 0], dataHenke[:, 1])
			f2 = np.interp(energy, dataHenke[:, 0], dataHenke[:, 2])
			f1sum += f1 * self.composition.indices[i]
			f2sum += f2 * self.composition.indices[i]

		lambda_ = hc/(energy/1000)
		f1sum /= np.sum(self.composition.indices)
		f2sum /= np.sum(self.composition.indices)

		n = 1 - self.atomic_density * r0 * 1e10 * lambda_**2 * f1sum/2/math.pi
		k = -self.atomic_density * r0 * 1e10 * lambda_**2 * f2sum/2/math.pi

		eps1 = n**2 - k**2
		eps2 = 2*n*k

		return energy, -eps2/(eps1**2 + eps2**2)

	def readhenke(self, filename):
		henke = np.loadtxt(filename + '.nff', skiprows = 1)
		return henke

	def calculateELF(self):
		ELF = (-1/self.epsilon).imag
		ELF[np.isnan(ELF)] = 1e-5
		self.ELF = ELF

	def calculateSurfaceELF(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculateDielectricFunction()
		eps_1 = self.epsilon.real
		eps_2 = self.epsilon.imag
		den = (eps_1**2 + eps_1 - eps_2**2)**2 + (2*eps_1*eps_2 + eps_2)**2
		enu = -eps_2*(2*eps_1 + 1)*((eps_1 - 1)**2 - eps_2**2)
		enu += 2*eps_2*(eps_1 - 1)*(eps_1*(eps_1 + 1) - eps_2**2)
		self.surfaceELF = enu/den

	def calculateDSEP(self, E0):
		self.calculateDielectricFunction()
		epsfraction = (self.epsilon - 1)**2 / (self.epsilon*(self.epsilon + 1))
		qs1 = np.sqrt(np.abs(self.q**2 - (self.eloss.reshape((-1,1)) + 0.5*self.q**2)**2 / (2.0 * E0)))  
		result = 2.0*epsfraction.imag *qs1 / self.q**3
		result[np.isinf(result)] = 0
		result[0,:] = 0
		self.surfaceELF = epsfraction.imag
		return result / (math.pi*E0)

	def calculateOpticalConstants(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculateDielectricFunction()
		n_complex = np.sqrt(self.epsilon)
		self.refractive_index = n_complex.real
		self.extinction_coefficient = n_complex.imag

	def calculateOpticalConstants_(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculateDielectricFunction()
		first = np.sqrt(self.epsilon.real ** 2 + self.epsilon.imag ** 2) / 2
		second = self.epsilon.real / 2
		self.refractive_index = np.sqrt(first + second)
		self.extinction_coefficient = np.sqrt(first - second)

	def plotELF(self, savefig = False, filename = None):
		if self.ELF is None or self.ELF.shape[0] != self.eloss.shape[0]:
			self.calculateELF()
		plt.figure()
		plt.plot(self.eloss, self.ELF, label='ELF')
		plt.xlabel('Energy loss $\omega$ (eV)')
		plt.ylabel('ELF')
		plt.title(f'{self.name} {self.oscillators.model}')
		# plt.xlim(0, 100)
		# plt.legend()
		plt.show()
		if savefig and filename:
			plt.savefig(filename, dpi=600)
	
	def calculateLiDiimfp_vs(self, E0, r, alpha, n_q=10, dE=0.5):
		old_eloss = self.eloss
		old_q = self.q
		old_E0 = E0

		if (self.Eg > 0):
			E0 = E0 - self.Eg
			if old_E0 <= 100:
				eloss = linspace(self.Eg, E0 - self.width_of_the_valence_band, dE)
			elif old_E0 <= 1000:
				range_1 = linspace(self.Eg, 100, dE)
				range_2 = linspace(101, E0 - self.width_of_the_valence_band, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(self.Eg, 100, dE)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, E0 - self.width_of_the_valence_band, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
		else:
			if old_E0 <= 100:
				eloss = linspace(1e-5, E0 - self.Ef, dE)
			elif old_E0 <= 1000:
				range_1 = linspace(1e-5, 100, dE)
				range_2 = linspace(101, E0 - self.Ef, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(1e-5, 100, dE)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, E0 - self.Ef, 100)
				eloss = np.concatenate((range_1, range_2, range_3))

		self.eloss = eloss
		rel_coef = ((1 + (E0/h2ev)/(c**2))**2) / (1 + (E0/h2ev)/(2*c**2))

		theta = np.linspace(0, math.pi/2, 10)
		phi = np.linspace(0, 2*math.pi, 10)
		v = math.sqrt(2*E0/h2ev)
		r /= (a0 * np.cos(alpha))
		
		q_minus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) - np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
		q_plus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) + np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
		q = np.linspace(q_minus, q_plus, 2**(n_q - 1), axis = 1)
		if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
			q[q == 0] = 0.01

		q_ = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))
		qsintheta = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))**2
		omegawave = (self.eloss/h2ev).reshape((-1,1,1,1)) - np.expand_dims(np.expand_dims(q,axis=2) * v * np.sin(theta.reshape((1,1,-1))), axis=3) * np.cos(phi.reshape((1,1,1,-1))) * np.sin(alpha)
		qz = np.expand_dims(q,axis=2) * np.cos(theta.reshape((1,1,-1)))
		coef = ( np.expand_dims(qsintheta,axis=3) * np.cos(np.expand_dims(qz, axis=3)*r*np.cos(alpha)) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )

		if (r >= 0):					
			
			self.q = q / a0
			self.calculateELF()
			integrand = self.ELF / q
			integrand[q == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q == 0.01] = 0
			bulk = 1/(math.pi * (E0/h2ev)) * np.trapz( integrand, q, axis = 1 ) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			self.q = q_ / a0
			self.calculateDielectricFunction()

			elf = (-1 / (self.epsilon + 1)).imag
			integrand = np.expand_dims(elf,axis=3)*coef
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			surf_outside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			
			coef = ( np.expand_dims(qsintheta,axis=3) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )
			coef_ = 2 * np.cos(omegawave * r / v) - np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha))
			
			integrand = np.expand_dims(elf,axis=3)*coef*coef_
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			surf_inside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)
			
			elf = (-1 / self.epsilon).imag
			integrand = np.expand_dims(elf,axis=3)*coef*coef_
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			bulk_reduced = 2*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			dsep = rel_coef * ( surf_inside + surf_outside )
			diimfp = rel_coef * bulk
			total = rel_coef * ( bulk - bulk_reduced + surf_inside + surf_outside )
			self.bulk_reduced = bulk_reduced
		else:
			self.q = q_ / a0
			self.calculateDielectricFunction()
			elf = (-1 / (self.epsilon + 1)).imag

			integrand = np.expand_dims(elf,axis=3)*coef
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			dsep = rel_coef * 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			diimfp = dsep
			total = dsep
			self.bulk_reduced = np.zeros_like(dsep)
		
		self.DIIMFP = diimfp
		self.totalDIIMFP = total
		
		self.DIIMFP_E = eloss
		self.DSEP = dsep
		self.E0 = old_E0
		self.eloss = old_eloss
		self.q = old_q
		self.sep = np.trapz(self.DSEP, eloss, axis=0) 

	def calculateLiDiimfp_sv(self, E0, r, alpha, n_q=10, dE=0.5):
		old_eloss = self.eloss
		old_q = self.q
		old_E0 = E0

		if (self.Eg > 0):
			E0 = E0 - self.Eg
			if old_E0 <= 100:
				eloss = linspace(self.Eg, E0 - self.width_of_the_valence_band, dE)
			elif old_E0 <= 1000:
				range_1 = linspace(self.Eg, 100, dE)
				range_2 = linspace(101, E0 - self.width_of_the_valence_band, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(self.Eg, 100, dE)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, E0 - self.width_of_the_valence_band, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
		else:
			if old_E0 <= 100:
				eloss = linspace(1e-5, E0 - self.Ef, dE)
			elif old_E0 <= 1000:
				range_1 = linspace(1e-5, 100, dE)
				range_2 = linspace(101, E0 - self.Ef, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(1e-5, 100, dE)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, E0 - self.Ef, 100)
				eloss = np.concatenate((range_1, range_2, range_3))

		self.eloss = eloss
		rel_coef = ((1 + (E0/h2ev)/(c**2))**2) / (1 + (E0/h2ev)/(2*c**2))

		theta = np.linspace(0, math.pi/2, 10)
		phi = np.linspace(0, 2*math.pi, 10)
		v = math.sqrt(2*E0/h2ev)
		r /= (a0 * np.cos(alpha))
		
		q_minus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) - np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
		q_plus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) + np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
		q = np.linspace(q_minus, q_plus, 2**(n_q - 1), axis = 1)
		if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
			q[q == 0] = 0.01

		q_ = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))
		qsintheta = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))**2
		omegawave = (self.eloss/h2ev).reshape((-1,1,1,1)) - np.expand_dims(np.expand_dims(q,axis=2) * v * np.sin(theta.reshape((1,1,-1))), axis=3) * np.cos(phi.reshape((1,1,1,-1))) * np.sin(alpha)
	
		if (r >= 0):					
			qz = np.expand_dims(q,axis=2) * np.cos(theta.reshape((1,1,-1)))
			coef = ( np.expand_dims(qsintheta,axis=3) * np.cos(np.expand_dims(qz, axis=3)*r*np.cos(alpha)) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )

			self.q = q / a0
			self.calculateELF()
			integrand = self.ELF / q
			integrand[q == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q == 0.01] = 0
			bulk = 1/(math.pi * (E0/h2ev)) * np.trapz( integrand, q, axis = 1 ) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			self.q = q_ / a0
			self.calculateDielectricFunction()

			elf = (-1 / self.epsilon).imag
			integrand = np.expand_dims(elf, axis=3) * coef
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			bulk_reduced = 2*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			elf = (-1 / (self.epsilon + 1)).imag
			integrand = np.expand_dims(elf,axis=3)*coef
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			surf_inside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)
			
			coef = ( np.expand_dims(qsintheta,axis=3) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )
			coef_ = 2 * np.cos(omegawave * r / v) - np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha))
			integrand = np.expand_dims(elf,axis=3)*coef*coef_
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q_ == 0.01] = 0
			surf_outside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			
			dsep = rel_coef * ( surf_inside + surf_outside )
			diimfp = rel_coef * bulk
			total = rel_coef * ( bulk - bulk_reduced + surf_inside + surf_outside )
			self.bulk_reduced = bulk_reduced
		else:
			self.q = q_ / a0
			self.calculateDielectricFunction()
			elf = (-1 / (self.epsilon + 1)).imag

			coef = ( np.expand_dims(qsintheta,axis=3) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )
			coef_ = 2 * np.cos(omegawave * r / v) - np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha))
			integrand = np.expand_dims(elf,axis=3)*coef*coef_
			dsep = rel_coef * 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2),q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			diimfp = dsep
			total = dsep
			self.bulk_reduced = np.zeros_like(dsep)
		
		self.DIIMFP = diimfp
		self.totalDIIMFP = total
		
		self.DIIMFP_E = eloss
		self.DSEP = dsep
		self.E0 = old_E0
		self.eloss = old_eloss
		self.q = old_q
		self.sep = np.trapz(self.DSEP, eloss, axis=0) 

	def calculateDIIMFP(self, E0, dE = 0.5, decdigs = 10, normalised = True):
		old_eloss = self.eloss
		old_q = self.q
		old_E0 = E0

		if (self.Eg > 0):
			E0 = E0 - self.Eg
			if old_E0 <= 100:
				eloss = linspace(self.Eg, E0 - self.width_of_the_valence_band, dE)
			elif old_E0 <= 1000:
				range_1 = linspace(self.Eg, 100, dE)
				range_2 = linspace(101, E0 - self.width_of_the_valence_band, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(self.Eg, 100, dE)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, E0 - self.width_of_the_valence_band, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
		else:
			if old_E0 <= 100:
				eloss = linspace(1e-5, E0 - self.Ef, dE)
			elif old_E0 <= 1000:
				range_1 = linspace(1e-5, 100, dE)
				range_2 = linspace(101, E0 - self.Ef, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(1e-5, 100, dE)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, E0 - self.Ef, 100)
				eloss = np.concatenate((range_1, range_2, range_3))

		self.eloss = eloss

		rel_coef = ((1 + (E0/h2ev)/(c**2))**2) / (1 + (E0/h2ev)/(2*c**2))

		if self.oscillators.alpha == 0 and self.oscillators.model != 'Mermin' and self.oscillators.model != 'MerminLL' and self.q_dependency is None:
			q_minus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) - np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
			q_plus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) + np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
			self.extendToHenke()
			int_limits = np.log(q_plus/q_minus)
			int_limits[np.isinf(int_limits)] = machine_eps
			interp_elf = np.interp(eloss, self.eloss_extended_to_Henke, self.ELF_extended_to_Henke)
			interp_elf[np.isnan(interp_elf)] = machine_eps
			diimfp = rel_coef * 1/(math.pi*(E0/h2ev)) * interp_elf * int_limits * (1/h2ev/a0)
		else:
			q_minus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) - np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
			q_plus = np.sqrt(E0/h2ev * (2 + E0/h2ev/(c**2))) + np.sqrt((E0/h2ev - self.eloss/h2ev) * (2 + (E0/h2ev - self.eloss/h2ev)/(c**2)))
			q = np.linspace(q_minus, q_plus, 2**(decdigs - 1), axis = 1)
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				q[q == 0] = 0.01
			self.q = q / a0

			# self.calculateSurfaceELF()
			# integrand_s = self.surfaceELF / q
			integrand_s = self.calculateDSEP(E0/h2ev)
			self.calculateELF()
			integrand = self.ELF / q
			integrand[q == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MerminLL'):
				integrand[q == 0.01] = 0
			diimfp = rel_coef * 1/(math.pi * (E0/h2ev)) * np.trapz( integrand, q, axis = 1 ) * (1/h2ev/a0)
			dsep = rel_coef * np.trapz( integrand_s, q, axis = 1 ) / h2ev

		diimfp[np.isnan(diimfp)] = 1e-5
		dsep[np.isnan(diimfp)] = 1e-5
		self.eloss = old_eloss
		self.q = old_q
		self.sep = np.trapz(dsep, eloss)

		if normalised:
			diimfp = diimfp / np.trapz(diimfp, eloss)
			dsep = dsep / np.trapz(dsep, eloss)
		
		self.DIIMFP = diimfp
		self.DIIMFP_E = eloss
		self.DSEP = dsep
		self.E0 = old_E0

	def plotDIIMFP(self, E0, decdigs = 10, normalised = True, savefig = False, filename = None):
		if self.DIIMFP is None or self.E0 != E0:
			self.calculateDIIMFP(E0, decdigs, normalised)
		plt.figure()
		plt.plot(self.DIIMFP_E, self.DIIMFP)
		plt.xlabel('Energy loss $\omega$ (eV)')
		if normalised:
			plt.ylabel('Normalised DIIMFP (eV$^{-1}$)')
		else:
			plt.ylabel('DIIMFP')
		plt.title(f'{self.name} {self.oscillators.model}')
		# plt.legend()
		# plt.xlim(0,100)
		plt.show()
		if savefig and filename:
			plt.savefig(filename, dpi=600)


	def calculateIMFP(self, energy, isMetal = True):
		if isMetal and self.Ef == 0:
			raise InputError("Please specify the value of the Fermi energy Ef")
		elif not isMetal and self.Eg == 0 and self.width_of_the_valence_band == 0:
			raise InputError("Please specify the values of the band gap Eg and the width of the valence band width_of_the_valence_band")
		imfp = np.zeros_like(energy)
		for i in range(energy.shape[0]):
			self.calculateDIIMFP(energy[i], 9, normalised = False)
			imfp[i] = 1/np.trapz(self.DIIMFP, self.DIIMFP_E)
		self.IMFP = imfp
		self.IMFP_E = energy

	def plotIMFP(self, energy, isMetal = True, savefig = False, filename = None):
		if self.IMFP is None or self.IMFP_E != energy:
			self.calculateIMFP(energy, isMetal)
		plt.figure()
		plt.plot(self.IMFP_E, self.IMFP)
		plt.xlabel('Energy (eV)')
		plt.ylabel('IMFP ($\mathrm{\AA}$)')
		plt.yscale('log')
		plt.xscale('log')
		plt.title(f'{self.name} {self.oscillators.model}')
		# plt.legend()
		plt.show()

	def get_sigma(self, lines, line, pattern):
		start = lines[line].find(pattern) + len(pattern)
		end = lines[line].find(' cm**2')
		return float(lines[line][start:end])*1e16

	def calculateElasticProperties(self, E0):
		self.E0 = E0
		fd = open('lub.in','w+')

		fd.write(f'IZ      {self.Z}         atomic number                               [none]\n')
		fd.write('MNUCL   3          rho_n (1=P, 2=U, 3=F, 4=Uu)                  [  3]\n')
		fd.write(f'NELEC   {self.Z}         number of bound electrons                    [ IZ]\n')
		fd.write('MELEC   3          rho_e (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)   [  4]\n')
		fd.write('MUFFIN  0          0=free atom, 1=muffin-tin model              [  0]\n')
		fd.write('RMUF    0          muffin-tin radius (cm)                  [measured]\n')
		fd.write('IELEC  -1          -1=electron, +1=positron                     [ -1]\n')
		fd.write('MEXCH   1          V_ex (0=none, 1=FM, 2=TF, 3=RT)              [  1]\n')
		fd.write('MCPOL   2          V_cp (0=none, 1=B, 2=LDA)                    [  0]\n')
		fd.write('VPOLA  -1          atomic polarizability (cm^3)            [measured]\n')
		fd.write('VPOLB  -1          b_pol parameter                          [default]\n')
		fd.write('MABS    1          W_abs (0=none, 1=LDA-I, 2=LDA-II)            [  0]\n')
		fd.write('VABSA   2.0        absorption-potential strength, Aabs      [default]\n')
		fd.write('VABSD  -1.0        energy gap DELTA (eV)                    [default]\n')
		fd.write('IHEF    2          high-E factorization (0=no, 1=yes, 2=Born)   [  1]\n')
		fd.write(f'EV      {round(self.E0)}     kinetic energy (eV)                         [none]\n')

		fd.close()

		# output = os.system('/Users/olgaridzel/Research/ESCal/src/MaterialDatabase/Data/Elsepa/elsepa-2020/elsepa-2020 < lub.in')
		x = subprocess.run('/Users/olgaridzel/Research/ESCal/src/MaterialDatabase/Data/Elsepa/elsepa-2020/elsepa-2020 < lub.in',shell=True,capture_output=True)

		with open('dcs_' + '{:1.3e}'.format(round(self.E0)).replace('.','p').replace('+0','0') + '.dat','r') as fd:
			self.sigma_el = self.get_sigma(fd.readlines(), 32, 'Total elastic cross section = ')
			self.EMFP = 1/(self.sigma_el*self.atomic_density)
			# sigma_tr_1 = get_sigma(lines, 33, '1st transport cross section = ')
			# sigma_tr_2 = get_sigma(lines, 34, '2nd transport cross section = ')
		
		data = np.loadtxt('dcs_' + '{:1.3e}'.format(round(self.E0)).replace('.','p').replace('+0','0') + '.dat', skiprows=44)
		self.DECS_deg = data[:,0]
		self.DECS_mu = data[:,1]
		self.DECS = data[:,2]
		self.NDECS = self.DECS / np.trapz(self.DECS, self.DECS_mu)

	def Legendre_mu(self,mu,m,L):
		smu = np.sqrt(1 - mu**2)

		N = mu.shape[0]
		m2 = m**2
		P = np.ones((L - m, N))
		if m > 0:
			im2 = linspace(2, 2*m, 2)
			sim2 = np.sqrt(np.prod(1 - 1 / im2))
			P[0,:] = sim2 * smu ** m

		sim1 = np.sqrt(2*m + 1)
		P[1,:] = sim1 * mu * P[0,:]

		for k in range(m + 2, L):
			i = k - m
			iskm = 1 / np.sqrt(k**2 - m2)
			k1 = (2*k - 1)*iskm
			k2 = np.sqrt((k-1)**2 - m2)*iskm
			P[i,:] = k1* mu * P[i-1,:] - k2*P[i-2,:]
		return P

	def calculateAngularMesh(self, N):
		[x, s] = special.roots_legendre(N-1)
		x = (np.concatenate((x, np.array([1]))) + 1) / 2
		self.mu_mesh = x[::-1]
		
	def calculateLegendreCoefficients(self, NLeg=100):
		N = max(2000, math.floor(NLeg*5))
		self.xl = np.zeros(NLeg+1)
		mu = np.cos(np.deg2rad(self.DECS_deg))
		[x,s] = special.roots_legendre(N)
		decs = np.interp(x, mu[::-1], self.DECS[::-1])
		decs /= np.trapz(decs, x)

		for l in range(NLeg+1):
			self.xl[l] = np.trapz(decs * special.lpmv(0, l, x), x)
		# self.xl = np.loadtxt('xl')

	def calculate(self, E0=1000, n_in=10, NLeg=100, mu_i=1, mu_o=0.5, phi=0):
		# N = max(2000, math.floor(NLeg*5))
		N = NLeg
		self.mu_i = mu_i
		self.mu_o = mu_o
		# if not self.E0 or self.E0 != E0:
		# self.calculateElasticProperties(E0)
		# self.calculateLegendreCoefficients(N)

		self.calculateIMFP(np.array([E0]))
		imfp = self.IMFP[0]
		self.TMFP = 1 / (1/imfp + 1/self.EMFP)
		self.albedo = self.TMFP / self.EMFP

		# psi = np.rad2deg(math.acos(mu_i*mu_o + math.sqrt(1 - mu_i**2)*math.sqrt(1 - mu_o**2)*math.cos(np.deg2rad(phi))))
		# print(psi)
		# M = math.floor(0.41*psi - 6e-3*psi**2 + 1.75e-10*psi**6 + 0.8)
		# print(M)
		M = 1
		norm = 1/2/math.pi
		self.calculateAngularMesh(81)

		mm = 1/self.mu_mesh + 1/self.mu_mesh.reshape((-1,1))
		B = norm/mm
		
		norm_leg = (2*linspace(0,N,1) + 1)/2
		self.R_l = np.zeros((self.mu_mesh.shape[0], self.mu_mesh.shape[0], n_in, N+1))

		for l in range(N+1):
			self.R_l[:,:,0,l] = -np.log(1 - self.albedo * self.xl[l]) * B
			for k in range(1,n_in):
				self.R_l[:,:,k,l] = (1 - self.albedo)**k / k * (1/(1 - self.albedo * self.xl[l])**k - 1) * B
				
		self.R_m = np.zeros((self.mu_mesh.shape[0], self.mu_mesh.shape[0], n_in, M))
		for m in range(M):
			for l in range(N+1):
				P = special.lpmv(m, l, self.mu_mesh)
				PlP = P * norm_leg[l+m] * P.reshape((-1,1))
				for k in range(n_in):
					self.R_m[:,:,k,m] += (-1)**(l + m) * self.R_l[:,:,k,l] * PlP

		self.R_m[np.isnan(self.R_m)] = 0
		self.R_m[np.isinf(self.R_m)] = 0
		self.calculateAngularDistribution(mu_i, n_in)
		self.calculatePartialIntensities(mu_o, n_in)

	def calculateAngularDistribution(self, mu_i, n_in):
		self.angular_distribution = np.zeros((self.mu_mesh.shape[0], n_in))
		ind = self.mu_mesh == mu_i

		for k in range(n_in):
			if any(ind):
				self.angular_distribution[:, k] = self.R_m[ind,:,k,0]*2*math.pi
			else:
				f = interpolate.interp1d(self.mu_mesh, self.R_m[:,:,k,0])
				self.angular_distribution[:, k] = f(mu_i)*2*math.pi

	def calculatePartialIntensities(self, mu_o, n_in):
		self.partial_intensities = np.zeros(n_in)
		ind = self.mu_mesh == mu_o

		for k in range(n_in):
			if any(ind):
				self.partial_intensities[k] = self.angular_distribution[ind,k]
			else:
				self.partial_intensities[k] = np.interp(mu_o, self.mu_mesh, self.angular_distribution[:,k])

	def calculateDiimfpConvolutions(self, n_in):
		if self.DIIMFP is None:
			raise Error("The diimfp has not been calculated yet.")
		de = self.DIIMFP_E[2] - self.DIIMFP_E[1]
		convolutions = np.zeros((self.DIIMFP.size, n_in+1))
		convolutions[0, 0] = 1/de    
		
		for k in range(1, n_in+1):
			convolutions[:,k] = conv(convolutions[:,k-1], self.DIIMFP, de)
		
		return convolutions

	def calculateDsepConvolutions(self, n_in):
		if self.DSEP is None:
			raise Error("The DSEP has not been calculated yet.")
		de = self.DIIMFP_E[2] - self.DIIMFP_E[1]
		convolutions = np.zeros((self.DSEP.size, n_in+1))
		convolutions[0, 0] = 1/de    
		
		for k in range(1, n_in+1):
			convolutions[:,k] = conv(convolutions[:,k-1], self.DSEP, de)
		
		return convolutions

	def fitElasticPeak(self, x_exp, y_exp):
		ind = np.logical_and(x_exp > x_exp[np.argmax(y_exp)] - 3, x_exp < x_exp[np.argmax(y_exp)] + 3)
		x = x_exp[ind]
		y = y_exp[ind]
		popt, pcov = optimize.curve_fit(gauss, x, y, [np.max(y), x[np.argmax(y)], 1])
		return popt

	def calculateEnergyDistribution(self, E0, theta_i, theta_o, n_in, x_exp, y_exp, dE=0.5, n_q=10):
		depth = np.array([-6,-4,-2,0,2,4,6])
		i = 0
		for r in depth:
			self.calculateLiDiimfp_vs(E0, r, theta_i, n_q, dE)

			if i == 0:
				dsep_i = np.zeros((len(depth),) + self.DSEP.shape)
				dsep_o = np.zeros((len(depth),) + self.DSEP.shape)
				total_i = np.zeros((len(depth),) + self.totalDIIMFP.shape)
				total_o = np.zeros((len(depth),) + self.totalDIIMFP.shape)

			dsep_i[i] = self.DSEP - self.bulk_reduced
			total_i[i] = self.totalDIIMFP

			self.calculateLiDiimfp_sv(E0, r, theta_o, n_q, dE)
			dsep_o[i] = self.DSEP - self.bulk_reduced
			total_o[i] = self.totalDIIMFP
			
			i += 1
		
		self.DIIMFP /= np.trapz(self.DIIMFP, self.DIIMFP_E)
		self.DIIMFP[np.isnan(self.DIIMFP)] = 1e-5
		
		self.SEP_i = np.trapz( np.trapz(total_i, self.DIIMFP_E,axis=1)[depth<=0], depth[depth<=0] / np.cos(theta_i))
		self.DSEP_i = np.trapz(dsep_i, depth / np.cos(theta_i), axis=0)	
		self.DSEP = self.DSEP_i / np.trapz(self.DSEP_i, self.DIIMFP_E)
		self.DSEP[np.isnan(self.DSEP)] = 1e-5	
		convs_s = self.calculateDsepConvolutions(n_in-1)
		self.partial_intensities_s_i = stats.poisson(self.SEP_i).pmf(range(n_in))
		self.energy_distribution_s_i = np.sum(convs_s*np.squeeze(self.partial_intensities_s_i),axis=1)
		convs_b = self.calculateDiimfpConvolutions(n_in-1)
		self.energy_distribution_b = np.sum(convs_b*np.squeeze(self.partial_intensities / self.partial_intensities[0]),axis=1)
		
		self.SEP_o = np.trapz( np.trapz(total_o, self.DIIMFP_E,axis=1)[depth<=0], depth[depth<=0] / np.cos(theta_o))	
		self.DSEP_o = np.trapz(dsep_o, depth / np.cos(theta_o), axis=0)
		self.DSEP = self.DSEP_o / np.trapz(self.DSEP_o, self.DIIMFP_E)
		self.DSEP[np.isnan(self.DSEP)] = 1e-5
		convs_s = self.calculateDsepConvolutions(n_in-1)
		self.partial_intensities_s_o = stats.poisson(self.SEP_o).pmf(range(n_in))
		self.energy_distribution_s_o = np.sum(convs_s*np.squeeze(self.partial_intensities_s_o),axis=1)
		
		extra = np.linspace(-10,-dE, round(10/dE))
		self.spectrum_E = np.concatenate((extra, self.DIIMFP_E))
		sb = conv(self.energy_distribution_s_i, self.energy_distribution_b, dE)
		sbs = np.concatenate((np.zeros_like(extra), conv(sb, self.energy_distribution_s_o, dE) ))
		coefs = self.fitElasticPeak(x_exp, y_exp)
		gaussian = gauss(linspace(-10,10,dE), coefs[0], 0, coefs[2])
		self.spectrum = conv(sbs, gaussian, dE)

	def writeOpticalData(self):
		self.calculateELF()
		self.calculateOpticalConstants()
		d = dict(E=np.round(self.eloss,2),n=np.round(self.refractive_index,2),k=np.round(self.extinction_coefficient,2),eps1=np.round(self.epsilon.real,2), eps2=np.round(self.epsilon.imag,2), elf=np.round(self.ELF,2))
		df = pd.DataFrame.from_dict(d, orient='index').transpose().fillna('')
		with open(f'{self.name}_{self.oscillators.model}_table_optical_data.csv', 'w') as tf:
			tf.write(df.to_csv(index=False))

class exp_data:
	def __init__(self):
		self.x_elf = []
		self.y_elf = []
		self.x_ndiimfp = []
		self.y_ndiimfp = []

class OptFit:

	def __init__(self, material, exp_data, E0, dE = 0.5, n_q = 10):
		if not isinstance(material, Material):
			raise InputError("The material must be of the type Material")
		if E0 == 0:
			raise InputError("E0 must be non-zero")
		self.material = material
		self.exp_data = exp_data
		self.E0 = E0
		self.dE = dE
		self.n_q = n_q
		self.count = 0
		
	def setBounds(self):
		osc_min_A = np.ones_like(self.material.oscillators.A) * 1e-10
		osc_min_gamma = np.ones_like(self.material.oscillators.gamma) * 0.025
		osc_min_omega = np.ones_like(self.material.oscillators.omega) * self.material.Eg
		
		if self.material.oscillators.model == 'Drude':
			osc_max_A = np.ones_like(self.material.oscillators.A) * 2e3
		else:
			osc_max_A = np.ones_like(self.material.oscillators.A)

		osc_max_gamma = np.ones_like(self.material.oscillators.gamma) * 100
		osc_max_omega = np.ones_like(self.material.oscillators.omega) * 500		

		if self.material.oscillators.model == 'DLL' or self.material.oscillators.model == 'MerminLL':
			osc_min_U = 0.0
			osc_max_U = 10.0
			self.lb = np.append( np.hstack((osc_min_A,osc_min_gamma,osc_min_omega)), osc_min_U )
			self.ub = np.append( np.hstack((osc_max_A,osc_max_gamma,osc_max_omega)), osc_max_U )
		elif self.material.oscillators.model == 'Mermin':
			self.lb = np.hstack((osc_min_A,osc_min_gamma,osc_min_omega))
			self.ub = np.hstack((osc_max_A,osc_max_gamma,osc_max_omega))
		else:
			osc_min_alpha = 0.0
			osc_max_alpha = 1.0
			self.lb = np.append( np.hstack((osc_min_A,osc_min_gamma,osc_min_omega)), osc_min_alpha )
			self.ub = np.append( np.hstack((osc_max_A,osc_max_gamma,osc_max_omega)), osc_max_alpha )

	def runOptimisation(self, diimfp_coef, elf_coef, maxeval = 1000, xtol_rel = 1e-6, isGlobal = False):
		print('Starting optimisation...')
		self.count = 0
		self.diimfp_coef = diimfp_coef
		self.elf_coef = elf_coef

		if isGlobal:
			opt_local = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2Vec(self.material)))
			opt_local.set_maxeval(maxeval)
			opt_local.set_xtol_rel(xtol_rel)
			opt_local.set_ftol_rel = 1e-20;

			opt = nlopt.opt(nlopt.AUGLAG, len(self.struct2Vec(self.material)))
			opt.set_local_optimizer(opt_local)
			opt.set_min_objective(self.objective_function)
			self.setBounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_Henke is None and self.material.ELF_Henke is None:
			# 		self.material.eloss_Henke, self.material.ELF_Henke = self.material.mopt()
			# 	self.material.electron_density_Henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_Henke / h2ev * self.material.ELF_Henke, self.material.eloss_Henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_Henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind)

			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)

			x = opt.optimize(self.struct2Vec(self.material))
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		else:
			opt = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2Vec(self.material)))
			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)
			opt.set_ftol_rel = 1e-20;
			if diimfp_coef == 0:
				opt.set_min_objective(self.objective_function_elf)
			elif elf_coef == 0:
				opt.set_min_objective(self.objective_function_ndiimfp)
			else:
				opt.set_min_objective(self.objective_function)
			self.setBounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_Henke is None and self.material.ELF_Henke is None:
			# 		self.material.eloss_Henke, self.material.ELF_Henke = self.material.mopt()
			# 	self.material.electron_density_Henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_Henke / h2ev * self.material.ELF_Henke, self.material.eloss_Henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_Henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind)

			x = opt.optimize(self.struct2Vec(self.material))
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		return x

	def runOptimisationSpec(self, mu_i, mu_o, n_in, maxeval=1000, xtol_rel=1e-6, isGlobal=False):
		print('Starting spec optimisation...')
		self.bar = tqdm(total=maxeval)
		self.count = 0
		self.mu_i = mu_i
		self.mu_o = mu_o
		self.n_in = n_in

		if isGlobal:
			opt_local = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2Vec(self.material)))
			opt_local.set_maxeval(maxeval)
			opt_local.set_xtol_rel(xtol_rel)
			opt_local.set_ftol_rel = 1e-20;

			opt = nlopt.opt(nlopt.AUGLAG, len(self.struct2Vec(self.material)))
			opt.set_local_optimizer(opt_local)
			opt.set_min_objective(self.objective_function_spec)
			self.setBounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_Henke is None and self.material.ELF_Henke is None:
			# 		self.material.eloss_Henke, self.material.ELF_Henke = self.material.mopt()
			# 	self.material.electron_density_Henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_Henke / h2ev * self.material.ELF_Henke, self.material.eloss_Henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_Henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind)

			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)

			self.material.calculateElasticProperties(self.E0)
			self.material.calculateLegendreCoefficients(200)
			x = opt.optimize(self.struct2Vec(self.material))
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		else:
			opt = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2Vec(self.material)))
			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)
			opt.set_ftol_rel = 1e-20
			opt.set_min_objective(self.objective_function_spec)
			self.setBounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_Henke is None and self.material.ELF_Henke is None:
			# 		self.material.eloss_Henke, self.material.ELF_Henke = self.material.mopt()
			# 	self.material.electron_density_Henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_Henke / h2ev * self.material.ELF_Henke, self.material.eloss_Henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_Henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_KK_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_refind)

			self.material.calculateElasticProperties(self.E0)
			self.material.calculateLegendreCoefficients(200)
			x = opt.optimize(self.struct2Vec(self.material))
			self.bar.close()
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		return x

	def struct2Vec(self, osc_struct):
		if osc_struct.oscillators.model == 'MerminLL' or osc_struct.oscillators.model == 'DLL':
			vec = np.append( np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega)), osc_struct.U )
		elif self.material.oscillators.model == 'Mermin':
			vec = np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega))
		else:
			vec = np.append( np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega)), osc_struct.oscillators.alpha )
		return vec

	def vec2Struct(self, osc_vec):
		if self.material.oscillators.model == 'Mermin':
			oscillators = np.split(osc_vec[:],3)
		else:
			oscillators = np.split(osc_vec[0:-1],3)
		material = copy.deepcopy(self.material)
		material.oscillators.A = oscillators[0]
		material.oscillators.gamma = oscillators[1]
		material.oscillators.omega = oscillators[2]

		if self.material.oscillators.model == 'MerminLL' or self.material.oscillators.model == 'DLL':
			material.U = osc_vec[-1]
		elif self.material.oscillators.model != 'Mermin':
			material.oscillators.alpha = osc_vec[-1]
			
		return material

	def objective_function_spec(self, osc_vec, grad):
		self.count += 1
		alpha_i = np.rad2deg(np.arccos(self.mu_i))
		alpha_o = np.rad2deg(np.arccos(self.mu_o))
		ind = np.logical_and(self.exp_data.x_spec > self.exp_data.x_spec[np.argmax(self.exp_data.y_spec)] - 3, self.exp_data.x_spec < self.exp_data.x_spec[np.argmax(self.exp_data.y_spec)] + 3)
		x = self.exp_data.x_spec[ind]
		y = self.exp_data.y_spec[ind]
		exp_area = np.trapz(y, x)
		material = self.vec2Struct(osc_vec)
		material.calculate(self.E0, self.n_in, 200, self.mu_i, self.mu_o)
		material.calculateEnergyDistribution(self.E0, alpha_i, alpha_o, self.n_in, self.exp_data.x_spec, self.exp_data.y_spec, self.dE, self.n_q)

		ind = self.exp_data.x_spec < self.exp_data.x_spec[np.argmax(self.exp_data.y_spec)] - 5
		spec_interp = np.interp(self.E0 - self.exp_data.x_spec[ind], material.spectrum_E - material.spectrum_E[np.argmax(material.spectrum)], material.spectrum)
		chi_squared = np.sum((self.exp_data.y_spec[ind] / exp_area - spec_interp / exp_area)**2 / self.exp_data.x_spec.size)
		# chi_squared = np.sum((self.exp_data.y_spec / exp_area - spec_interp / exp_area)**2)

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])

		self.bar.update(1)
		time.sleep(1)
		return chi_squared

	def objective_function_ndiimfp(self, osc_vec, grad):
		self.count += 1
		material = self.vec2Struct(osc_vec)
		material.calculateDIIMFP(self.E0, self.dE, self.n_q)
		diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.DIIMFP_E, material.DIIMFP)
		chi_squared = np.sum((self.exp_data.y_ndiimfp - diimfp_interp)**2 / self.exp_data.x_ndiimfp.size)

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])
		return chi_squared

	def objective_function_elf(self, osc_vec, grad):
		self.count += 1
		material = self.vec2Struct(osc_vec)
		material.calculateELF()
		elf_interp = np.interp(self.exp_data.x_elf, material.eloss, material.ELF)
		chi_squared = np.sum((self.exp_data.y_elf - elf_interp)**2 / self.exp_data.x_elf.size)

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])
		return chi_squared

	def objective_function(self, osc_vec, grad):
		self.count += 1
		material = self.vec2Struct(osc_vec)
		material.calculateDIIMFP(self.E0, self.dE, self.n_q)
		diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.DIIMFP_E, material.DIIMFP)

		if material.oscillators.alpha == 0:
			elf_interp = np.interp(self.exp_data.x_elf, material.eloss_extended_to_Henke, material.ELF_extended_to_Henke)
		else:
			elf_interp = np.interp(self.exp_data.x_elf, material.DIIMFP_E, material.ELF[:,0])
		ind_ndiimfp = self.exp_data.y_ndiimfp >= 0
		ind_elf = self.exp_data.y_elf >= 0

		chi_squared = self.diimfp_coef*np.sqrt(np.sum((self.exp_data.y_ndiimfp[ind_ndiimfp] - diimfp_interp[ind_ndiimfp])**2 / len(self.exp_data.y_ndiimfp[ind_ndiimfp]))) + \
						self.elf_coef*np.sqrt(np.sum((self.exp_data.y_elf[ind_elf] - elf_interp[ind_elf])**2) / len(self.exp_data.y_elf[ind_elf]))

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])
		return chi_squared

	def constraint_function(self, osc_vec, grad):
		material = self.vec2Struct(osc_vec)
		material.convert2au()
		if material.oscillators.model == 'Drude':
			cf = material.electron_density * wpc / np.sum(material.oscillators.A)
		else:
			cf = (1 - 1 / material.static_refractive_index ** 2) / np.sum(material.oscillators.A)
		val = np.fabs(cf - 1)

		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_refind(self, osc_vec, grad):
		material = self.vec2Struct(osc_vec)
		material.convert2au()
		if material.oscillators.model == 'Drude':
			cf = 1
		else:
			bethe_sum = np.sum((math.pi / 2) * material.oscillators.A * material.oscillators.omega ** 2)
			bethe_value = 2 * math.pi ** 2 * material.electron_density * a0 ** 3
			cf = np.sqrt(bethe_sum / bethe_value)

		val = np.fabs(cf - 1)

		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_henke(self, osc_vec, grad):
		global iteration
		material = self.vec2Struct(osc_vec)
		material.convert2au()

		if material.oscillators.model == 'Drude':
			cf = self.material.electron_density_Henke * 4 * math.pi / np.sum(material.oscillators.A)
		else:
			cf = (1 - 1 / self.material.static_refractive_index ** 2) / np.sum(material.oscillators.A)

		val = np.fabs(cf - 1)
		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_refind_henke(self, osc_vec, grad):
		global iteration
		material = self.vec2Struct(osc_vec)
		material.convert2au()
		if material.oscillators.model == 'Drude':
			cf = 1
		else:
			bethe_sum = np.sum((math.pi / 2) * material.oscillators.A * material.oscillators.omega ** 2)
			bethe_value = 2 * math.pi ** 2 * self.material.electron_density_Henke
			cf = np.sqrt(bethe_sum / bethe_value)
		val = np.fabs(cf - 1)
		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val