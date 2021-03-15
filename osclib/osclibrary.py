from typing import List, Tuple
import numpy as np
import math

hc = 12.3981756608  # planck constant times velocity of light keV Angstr
r0 = 2.8179403227e-15
h2ev = 27.21184  # Hartree, converts Hartree to eV
a0 = 0.529177  # Bohr Radius in Angstroem


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
		if len(elements) != len(indices):
			raise InputError("The size of elements and indices should be the same!")
		self.elements = elements
		self.indices = indices


class Osc:
	def __init__(self, model, name, composition, omega, A, gamma):
		self.name = name
		self.composition = Composition(composition.elements, composition.indices)
		self.model = model
        self.omega = omega
        self.A = A
        self.gamma = gamma
        self.alpha = 0.0
        self.Eg = None
        self.Ef = None
        self.eps_b = 1.0
        self.eloss = [i for i in range(100)]
        self.q = 0.0
        self.vb = None
        self.na = 0.0

    def convert2au(self):
        if self.model == 'Drude':
            self.A = self.A/h2ev/h2ev
        self.gamma = self.gamma/h2ev
        self.omega = self.omega/h2ev
        self.Ef = self.Ef/h2ev
        self.eloss = self.eloss/h2ev
        self.q = self.q*a0
        if (self.Eg):
            self.Eg = self.Eg/h2ev
        if (self.vb):
            self.vb = self.vb/h2ev

    def convert2ru(self):
        if self.model == 'Drude':
            self.A = self.A*h2ev*h2ev
        self.gamma = self.gamma*h2ev
        self.omega = self.omega*h2ev
        self.Ef = self.Ef*h2ev
        self.eloss = self.eloss*h2ev
        self.q = self.q/a0
        if (self.Eg):
            self.Eg = self.Eg*h2ev

class Drude(Osc):
	model = 'Drude'
	def __init__(self, name, composition, omega, A, gamma):
		super().__init__(model, name, composition, omega, A, gamma)

	def calculateDielectricFunction(self):
		self.convert2au()
		if len(self.q.shape) == 1:
			eps_real = np.squeeze(
				self.eps_b*np.ones((self.eloss.shape[0], self.q.shape[0])))
			eps_imag = np.squeeze(np.zeros((self.eloss.shape[0], self.q.shape[0])))
		else:
			eps_real = self.eps_b*np.ones((self.eloss.shape[0], self.q.shape[1]))
			eps_imag = np.zeros((self.eloss.shape[0], self.q.shape[1]))

		epsilon = np.zeros_like(eps_real, dtype=complex)

		for i in range(len(self.A)):
			epsDrude_real, epsDrude_imag = self.calculateOneOscillator(
				self.omega[i], self.gamma[i], self.alpha)
			eps_real -= self.A[i]*epsDrude_real
			eps_imag += self.A[i]*epsDrude_imag

		epsilon.real = eps_real
		epsilon.imag = eps_imag
		self.convert2ru()
		return epsilon

	def calculateOneOscillator(self, omega0, gamma, alpha):
		w_at_q = omega0 + 0.5 * alpha * self.q**2
		if len(q.shape) == 1:
			num_q = self.q.shape[0]
		else:
			num_q = self.q.shape[1]

		omega = np.squeeze(np.array([self.eloss, ]*num_q).transpose())

		mm = omega**2 - w_at_q**2
		divisor = mm**2 + omega**2 * gamma**2

		eps_real = mm / divisor
		eps_imag = omega*gamma / divisor

		return eps_real, eps_imag

class DrudeLindhard(Osc):
	model = 'DrudeLindhard'
	def __init__(self, name, composition, omega, A, gamma):
		super().__init__(model, name, composition, omega, A, gamma)

	def calculateDielectricFunction(self):
		self.convert2au()
		if len(self.q.shape) == 1:
			epsilon = np.squeeze(
				np.zeros((self.eloss.shape[0], self.q.shape[0]), dtype=complex))
			sum_oneover_eps = np.squeeze(
				np.zeros((self.eloss.shape[0], self.q.shape[0]), dtype=complex))
			oneover_eps = np.squeeze(
				np.zeros((self.eloss.shape[0], self.q.shape[0]), dtype=complex))
		else:
			epsilon = np.zeros((self.eloss.shape[0], self.q.shape[1]), dtype=complex)
			sum_oneover_eps = np.zeros(
				(self.eloss.shape[0], self.q.shape[1]), dtype=complex)
			oneover_eps = np.zeros(
				(self.eloss.shape[0], self.q.shape[1]), dtype=complex)

		for i in range(len(self.A)):
			oneover_eps = self.calculateOneOscillator(self.omega[i], self.gamma[i], self.alpha)
			sum_oneover_eps += osc.A[i] * (oneover_eps - complex(1))

		sum_oneover_eps += complex(1)
		epsilon = complex(1) / sum_oneover_eps
		self.convert2ru()
		return epsilon

	def calculateOneOscillator(omega0, gamma, alpha):
		w_at_q = omega0 + 0.5 * alpha * self.q**2
		if len(q.shape) == 1:
			num_q = self.q.shape[0]
		else:
			num_q = self.q.shape[1]

		omega = np.squeeze(np.array([self.eloss, ]*num_q).transpose())

		mm = omega**2 - w_at_q**2
		divisor = mm**2 + omega**2 * gamma**2

		oneover_eps_real = 1.0 + omega0**2 * mm / divisor
		oneover_eps_imag = -omega0**2 * omega * gamma / divisor

		oneover_eps = np.squeeze(np.apply_along_axis(lambda args: [complex(
			*args)], 0, np.array([oneover_eps_real, oneover_eps_imag])))

		return oneover_eps


def linspace(start, stop, step=1.):
    num = int((stop - start) / step + 1)
    return np.linspace(start, stop, num)

def diimfp(osc: Osc, E0, decdigs):
    old_eloss = osc.eloss
    eloss = linspace(machine_eps, E0, 0.1)
    osc.eloss = eloss

    if osc.alpha == 0:
        q_minus = np.sqrt(2*E0/h2ev) - np.sqrt(2*(E0/h2ev-osc.eloss/h2ev))
        q_plus = np.sqrt(2*E0/h2ev) + np.sqrt(2*(E0/h2ev-osc.eloss/h2ev))
        eps = eps_sum(osc)
        eps[np.isnan(eps)] = machine_eps
        energy_henke, elf_henke = mopt(osc.composition, osc.na)
        ind_henke = energy_henke > 100
        ind = osc.eloss <= 100
        eloss_total = np.concatenate((osc.eloss[ind], energy_henke[ind_henke]))
        elf_total = np.interp(osc.eloss, eloss_total, np.concatenate(
            ((-1/eps[ind]).imag, elf_henke[ind_henke])))
        int_limits = np.log(q_plus/q_minus)
        int_limits[np.isinf(int_limits)] = machine_eps
        w = 1/(math.pi*(E0/h2ev)) * elf_total * int_limits * (1/h2ev/a0)
    else:
        w = np.zeros_like(osc.eloss)
        q_minus = np.log(np.sqrt(2*E0/h2ev) -
                         np.sqrt(2*(E0/h2ev-osc.eloss/h2ev)))
        q_plus = np.log(np.sqrt(2*E0/h2ev) +
                        np.sqrt(2*(E0/h2ev-osc.eloss/h2ev)))
        q = np.linspace(q_minus, q_plus, 2 ^ (decdigs-1), axis=1)
        osc.q = np.exp(q)/a0
        eps = eps_sum(osc)
        eps[np.isnan(eps)] = machine_eps
        for i in range(osc.eloss.shape[0]):
            w[i] = 1/(math.pi*(E0/h2ev)) * \
                np.trapz((-1/eps[i, :]).imag, q[i, :])*(1/h2ev/a0)

    w[np.isnan(w)] = machine_eps
    osc.eloss = old_eloss

    return eloss, w


def imfp(osc, energy, isMetal=False):
    lambda_in = np.zeros_like(energy)
    for i in range(energy.shape[0]):
        eloss, w = diimfp(osc, energy[i], 12)
        eloss_step = 0.5
        if isMetal:
            interp_eloss = linspace(
                machine_eps, energy[i] - osc.Ef, eloss_step)
        else:
            interp_eloss = linspace(
                machine_eps, energy[i] - (osc.Eg + osc.vb), eloss_step)
        interp_w = np.interp(interp_eloss, eloss, w)
        interp_w[np.isnan(interp_w)] = machine_eps

        lambda_in[i] = 1/np.trapz(interp_w, interp_eloss)

    return lambda_in
