import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
import nlopt
import copy
import pandas as pd

hc = 12.3981756608  # planck constant times velocity of light keV Angstr
r0 = 2.8179403227e-15
h2ev = 27.21184  # Hartree, converts Hartree to eV
a0 = 0.529177  # Bohr Radius in Angstroem
machine_eps = np.finfo('float64').eps
wpc = 1.8621440006


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
        self.surfaceELF = None
        # self.epsilon = None
        self.DIIMFP = None
        self.DIIMFP_E = None
        self.E0 = None
        self.IMFP = None
        self.IMFP_E = None
        self.q_dependency = None
        self.q = np.array(q)
        self.size_q = self.q.size

    @property
    def epsilon(self):
        self.calculateDielectricFunction()
        return self._epsilon

    def calculateDielectricFunction(self):
        if self.oscillators.model == 'Drude':
            self._epsilon = self.calculateDrudeDielectricFunction()
        elif self.oscillators.model == 'DrudeLindhard':
            self._epsilon = self.calculateDLDielectricFunction()
        elif self.oscillators.model == 'DLL':
            self._epsilon = self.calculateDLLDielectricFunction()
        elif self.oscillators.model == 'Mermin':
            self._epsilon = self.calculateMerminDielectricFunction()
        elif self.oscillators.model == 'MerminLL':
            self._epsilon = self.calculateMerminLLDielectricFunction()
        else:
            raise InputError("Invalid model name. The valid model names are: Drude, DrudeLindhard, Mermin and MerminLL")

    def calculateDrudeDielectricFunction(self):
        self.convert2au()
        eps_real = self.oscillators.eps_b * \
            np.squeeze(np.ones((self.eloss.size, self.size_q)))
        eps_imag = np.squeeze(np.zeros((self.eloss.size, self.size_q)))
        epsilon = np.zeros_like(eps_real, dtype=complex)

        for i in range(len(self.oscillators.A)):
            epsDrude_real, epsDrude_imag = self.calculateDrudeOscillator(
                self.oscillators.omega[i], self.oscillators.gamma[i], self.oscillators.alpha)
            eps_real -= self.oscillators.A[i] * epsDrude_real
            eps_imag += self.oscillators.A[i] * epsDrude_imag

        epsilon.real = eps_real
        epsilon.imag = eps_imag
        self.convert2ru()
        return epsilon

    def calculateDrudeOscillator(self, omega0, gamma, alpha):
        if not self.q_dependency is None:
            w_at_q = omega0 + (self.q_dependency(self.q / a0)/h2ev - self.q_dependency(0)/h2ev)
        else:
            w_at_q = omega0 + 0.5 * alpha * self.q**2

        omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())

        mm = omega**2 - w_at_q**2
        divisor = mm**2 + omega**2 * gamma**2

        eps_real = mm / divisor
        eps_imag = omega*gamma / divisor

        return eps_real, eps_imag

    def calculateDLLDielectricFunction(self):
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
            epsDL = self.calculateDLLOscillator(
                self.oscillators.omega[i], self.oscillators.gamma[i], self.oscillators.alpha)
            oneovereps += self.oscillators.A[i] * (complex(1) / epsDL - complex(1))
        oneovereps += complex(1)
        epsilon = complex(1) / oneovereps
        self.convert2ru()
        return epsilon

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
        self.convert2ru()
        return epsilon

    def calculateDLLOscillator(self, omega0, gamma, alpha):
        omegaDL = np.sqrt(omega0**2 + self.U**2)
        ratioInt = (omega0 / omegaDL)**2
        # z1 = ratioInt * self.calculateDLOscillator(omegaDL, gamma, 0.0) + (1.0 - ratioInt) * complex(1)
        z1 = ratioInt * self.calculateDLOscillator(omegaDL, gamma, alpha) + (1.0 - ratioInt) * complex(1)
        return complex(1) / z1


    def calculateDLOscillator(self, omega0, gamma, alpha):
        if not self.q_dependency is None:
            w_at_q = omega0 - self.q_dependency(0)/h2ev + self.q_dependency(self.q / a0)/h2ev
        else:
            w_at_q = omega0 + 0.5 * alpha * self.q**2

        omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())

        mm = omega**2 - w_at_q**2
        divisor = mm**2 + omega**2 * gamma**2

        oneover_eps_real = 1.0 + omega0**2 * mm / divisor
        oneover_eps_imag = -omega0**2 * omega * gamma / divisor

        oneover_eps = np.squeeze(np.apply_along_axis(lambda args: [complex(
            *args)], 0, np.array([oneover_eps_real, oneover_eps_imag])))

        return oneover_eps

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
        epsilon = complex(1) / oneovereps
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
            theta = atan2(DeltaSquare.imag, DeltaSquare.real)
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
        self.extendToHenke()
        div = self.ELF_extended_to_Henke / self.eloss_extended_to_Henke
        div[np.isnan(div)] = machine_eps
        kksum = 2 / math.pi * np.trapz(div, self.eloss_extended_to_Henke) + 1/self.static_refractive_index**2
        self.q = old_q
        return kksum

    def extendToHenke(self):
        self.calculateELF()
        energy_henke, elf_henke = self.mopt()
        ind_henke = energy_henke > 100
        ind = self.eloss <= 100
        self.eloss_extended_to_Henke = np.concatenate((self.eloss[ind], energy_henke[ind_henke]))
        self.ELF_extended_to_Henke = np.concatenate((self.ELF[ind], elf_henke[ind_henke]))
    
    def mopt(self):
        if self.atomic_density is None:
            raise InputError("Please specify the value of the atomic density atomic_density")
        numberOfElements = len(self.composition.elements)
        f1sum = 0
        f2sum = 0

        for i in range(numberOfElements):
            dataHenke = self.readhenke(
                self.xraypath + self.composition.elements[i])
            f1sum += dataHenke[:, 1] * self.composition.indices[i]
            f2sum += dataHenke[:, 2] * self.composition.indices[i]

        lambd = hc/(dataHenke[:, 0]/1000)
        f1sum /= np.sum(self.composition.indices)
        f2sum /= np.sum(self.composition.indices)

        n = 1 - self.atomic_density * r0 * 1e10 * lambd**2 * f1sum/2/math.pi
        k = -self.atomic_density * r0 * 1e10 * lambd**2 * f2sum/2/math.pi

        eps1 = n**2 - k**2
        eps2 = 2*n*k
        return dataHenke[:, 0], -eps2/(eps1**2 + eps2**2)

    def readhenke(self, filename):
        i = 0
        f = open(filename + '.', 'rt')
        for line in f:
            if line.startswith('#'):
                continue
            i += 1
        f.close()
        henke = np.zeros((i, 3))
        i = 0
        f = open(filename + '.', 'rt')
        for line in f:
            if line.startswith('#'):
                continue
            henke[i, :] = line.split()
            i += 1
        f.close()
        return henke

    def calculateELF(self):
        # if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
        #     self.calculateDielectricFunction()
        ELF = (-1/self.epsilon).imag
        ELF[np.isnan(ELF)] = machine_eps
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

    def calculateOpticalConstants(self):
        if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
            self.calculateDielectricFunction()
        n_complex = np.sqrt(self.epsilon)
        self.refractive_index = n_complex.real
        self.extinction_coefficient = n_complex.imag

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

    def calculateDIIMFP(self, E0, decdigs = 10, normalised = True):
        old_eloss = self.eloss
        old_q = self.q
        old_size_q = self.size_q
        eloss = linspace(machine_eps, E0, 0.1)
        self.eloss = eloss
        diimfp = np.zeros_like(self.eloss)

        if self.oscillators.alpha == 0 and self.oscillators.model != 'Mermin' and self.oscillators.model != 'MerminLL' and self.q_dependency is None:
            q_minus = np.sqrt(2 * E0/h2ev) - np.sqrt(2 *
                                                     (E0/h2ev - self.eloss/h2ev))
            q_plus = np.sqrt(2 * E0/h2ev) + np.sqrt(2 *
                                                    (E0/h2ev - self.eloss/h2ev))
            self.extendToHenke()
            int_limits = np.log(q_plus/q_minus)
            int_limits[np.isinf(int_limits)] = machine_eps
            ind = self.eloss_extended_to_Henke <= E0
            interp_elf = np.interp(eloss, self.eloss_extended_to_Henke, self.ELF_extended_to_Henke)
            interp_elf[np.isnan(interp_elf)] = machine_eps
            diimfp = 1/(math.pi*(E0/h2ev)) * interp_elf * int_limits * (1/h2ev/a0)
        else:
            q_minus = np.log( np.sqrt(2*E0/h2ev) - np.sqrt(2 * (E0/h2ev - self.eloss/h2ev)))
            q_plus = np.log( np.sqrt(2*E0/h2ev) + np.sqrt(2 * (E0/h2ev - self.eloss/h2ev)))
            q = np.linspace(q_minus, q_plus, 2**(decdigs - 1), axis = 1)
            self.size_q = q.shape[1]
            self.q = np.exp(q) / a0
            self.calculateDielectricFunction()
            diimfp = 1/(math.pi * (E0/h2ev)) * np.trapz( (-1/self.epsilon).imag, q, axis = 1 ) * (1/h2ev/a0)

        diimfp[np.isnan(diimfp)] = machine_eps
        self.eloss = old_eloss
        self.q = old_q
        self.size_q = old_size_q
        if normalised:
            diimfp = diimfp / np.trapz(diimfp, eloss)
        self.DIIMFP = diimfp
        self.DIIMFP_E = eloss
        self.E0 = E0

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
            self.calculateDIIMFP(energy[i], 12, normalised = False)
            eloss_step = 0.1
            if isMetal:
                interp_eloss = linspace(
                    machine_eps, energy[i] - self.Ef, eloss_step)
            else:
                interp_eloss = linspace(
                    machine_eps, energy[i] - (self.Eg + self.width_of_the_valence_band), eloss_step)
            interp_diimfp = np.interp(interp_eloss, self.DIIMFP_E, self.DIIMFP)
            interp_diimfp[np.isnan(interp_diimfp)] = machine_eps
            imfp[i] = 1/np.trapz(interp_diimfp, interp_eloss)
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

    def writeOpticalData(self):
        self.calculateELF()
        self.calculateOpticalConstants()
        d = dict(E=np.round(self.eloss,2),n=np.round(self.refractive_index,2),k=np.round(self.extinction_coefficient,2),eps1=np.round(self.epsilon.real,2), eps2=np.round(self.epsilon.imag,2), elf=np.round(self.ELF,2))
        df = pd.DataFrame.from_dict(d, orient='index').transpose().fillna('')
        with open(f'{self.name}_{self.oscillators.model}_table_optical_data.csv', 'w') as tf:
            tf.write(df.to_csv(index=False))

class OptFit:
    
    def __init__(self, material, x_exp, y_exp, E0):
        if not isinstance(material, Material):
            raise InputError("The material must be of the type Material")
        if E0 == 0:
            raise InputError("E0 must be non-zero")
        self.material = material
        self.x_exp = x_exp
        self.y_exp = y_exp
        self.E0 = E0
        
    def setBounds(self):
        osc_min_A = np.ones_like(self.material.oscillators.A) * 1e-10
        osc_min_gamma = np.ones_like(self.material.oscillators.gamma) * 0.25
        osc_min_omega = np.ones_like(self.material.oscillators.omega) * self.material.Eg

        if self.material.oscillators.model == 'Drude':
            osc_max_A = np.ones_like(self.material.oscillators.A) * 1e3
        else:
            osc_max_A = np.ones_like(self.material.oscillators.A)

        osc_max_gamma = np.ones_like(self.material.oscillators.gamma) * 100
        osc_max_omega = np.ones_like(self.material.oscillators.omega) * self.x_exp[-1]

        self.lb = np.hstack((osc_min_A,osc_min_gamma,osc_min_omega))
        self.ub = np.hstack((osc_max_A,osc_max_gamma,osc_max_omega))
    
    def runOptimisation(self, maxeval = 1000, xtol_rel = 1e-6):
        opt = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2Vec(self.material)))
        opt.set_min_objective(self.objective_function)
        self.setBounds()
        opt.set_lower_bounds(self.lb)
        opt.set_upper_bounds(self.ub)
        opt.add_inequality_constraint(self.constraint_function)
        opt.set_maxeval(maxeval)
        opt.set_xtol_rel(xtol_rel)
        return opt.optimize(self.struct2Vec(self.material))
    
    def struct2Vec(self, osc_struct):
        return np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega))
    
    def vec2Struct(self, osc_vec):
        oscillators = np.split(osc_vec,3)
        material = copy.deepcopy(self.material)
        material.oscillators.A = oscillators[0]
        material.oscillators.gamma = oscillators[1]
        material.oscillators.omega = oscillators[2]
        return material
    
    def objective_function(self, osc_vec, grad):
        material = self.vec2Struct(osc_vec)
        material.calculateDIIMFP(self.E0, 10)
        diimfp_interp = np.interp(self.x_exp, material.DIIMFP_E, material.DIIMFP)
        chi_squared = np.sum((self.y_exp - diimfp_interp)**2)

        if grad.size > 0:
            grad = np.array([0,0.5/chi_squared])
        return chi_squared
    
    def constraint_function(self, osc_vec, grad):
        material = self.vec2Struct(osc_vec)
        material.convert2au()
        if material.oscillators.model == 'Drude':
            cf = material.electron_density * wpc / np.sum(material.oscillators.A)
        else:
            cf = (1 - 1/material.static_refractive_index**2) / np.sum(material.oscillators.A)
        val = np.fabs(cf-1)

        if grad.size > 0:
            grad = np.array([0,0.5/val])

        return val