from typing import List, Tuple
import numpy as np
import math

hc = 12.3981756608  # planck constant times velocity of light keV Angstr
r0 = 2.8179403227e-15
h2ev = 27.21184  # Hartree, converts Hartree to eV
a0 = 0.529177  # Bohr Radius in Angstroem


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


class Osc:
    def __init__(self, model, name, composition, omega, A, gamma, eloss, q):
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
        self.name = name
        self.composition = composition
        self.model = model
        self.omega = np.array(omega)
        self.A = np.array(A)
        self.gamma = np.array(gamma)
        self.alpha = 0.0
        self.Eg = 0.0
        self.Ef = 0.0
        self.eps_b = 1.0
        self.eloss = np.array(eloss)
        self.vb = 0.0
        self.na = 0.0
        if isinstance(q, list):
            self.size_q = len(q)
            self.q = np.array(q)
        else:
            self.size_q = 1
            self.q = q

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

    def __init__(self, omega, A, gamma, eloss=linspace(0, 100, 1), q=0.0, name=None, composition=None):
        super().__init__(self.model, name, composition, omega, A, gamma, eloss, q)

    def calculateDielectricFunction(self):
        self.convert2au()
        eps_real = self.eps_b * \
            np.squeeze(np.ones((self.eloss.shape[0], self.size_q)))
        eps_imag = np.squeeze(np.zeros((self.eloss.shape[0], self.size_q)))
        epsilon = np.zeros_like(eps_real, dtype=complex)

        for i in range(len(self.A)):
            epsDrude_real, epsDrude_imag = self.calculateOneOscillator(
                self.omega[i], self.gamma[i], self.alpha)
            eps_real -= self.A[i]*epsDrude_real
            eps_imag += self.A[i]*epsDrude_imag

        epsilon.real = eps_real
        epsilon.imag = eps_imag
        self.convert2ru()
        self.epsilon = epsilon

    def calculateOneOscillator(self, omega0, gamma, alpha):
        w_at_q = omega0 + 0.5 * alpha * self.q**2

        omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())

        mm = omega**2 - w_at_q**2
        divisor = mm**2 + omega**2 * gamma**2

        eps_real = mm / divisor
        eps_imag = omega*gamma / divisor

        return eps_real, eps_imag


class DrudeLindhard(Osc):
    model = 'DrudeLindhard'

    def __init__(self, omega, A, gamma, eloss=linspace(0, 100, 1), q=0.0, name=None, composition=None):
        super().__init__(self.model, name, composition, omega, A, gamma, eloss, q)

    def calculateDielectricFunction(self):
        self.convert2au()
        epsilon = np.squeeze(
            np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
        sum_oneover_eps = np.squeeze(
            np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
        oneover_eps = np.squeeze(
            np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

        for i in range(len(self.A)):
            oneover_eps = self.calculateOneOscillator(
                self.omega[i], self.gamma[i], self.alpha)
            sum_oneover_eps += self.A[i] * (oneover_eps - complex(1))

        sum_oneover_eps += complex(1)
        epsilon = complex(1) / sum_oneover_eps
        self.convert2ru()
        self.epsilon = epsilon

    def calculateOneOscillator(self, omega0, gamma, alpha):
        w_at_q = omega0 + 0.5 * alpha * self.q**2

        omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())

        mm = omega**2 - w_at_q**2
        divisor = mm**2 + omega**2 * gamma**2

        oneover_eps_real = 1.0 + omega0**2 * mm / divisor
        oneover_eps_imag = -omega0**2 * omega * gamma / divisor

        oneover_eps = np.squeeze(np.apply_along_axis(lambda args: [complex(
            *args)], 0, np.array([oneover_eps_real, oneover_eps_imag])))

        return oneover_eps


class InelasticProperties:
    osc: Osc

    def diimfp(osc, E0, decdigs):
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
            eloss_total = np.concatenate(
                (osc.eloss[ind], energy_henke[ind_henke]))
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
