import numpy as np
from scipy import optimize, constants
import matplotlib.pyplot as plt
from tdwg.lib.mode_solver import *

def d_to_M(f, d):
    # assuming the thin lens equation and a measurement of d = i + o
    # returns the two possible imaging solutions (magnification and demagnification)
    i1 = d/2 + np.sqrt(d**2 / 4 - d * f)
    i2 = d/2 - np.sqrt(d**2 / 4 - d * f)

    o1 = d - i1
    o2 = d - i2

    M_magnifying = -i1 / o1
    M_demagnifying = -i2 / o2
    return M_magnifying, M_demagnifying

def chi3_to_n2(chi3, n0):
    # formula from https://www.rp-photonics.com/nonlinear_index.html
    return 3 * chi3 / 4 / constants.epsilon_0 / constants.c / n0 / np.real(n0)

def n2_to_chi3(n2, n0):
    return n2 / 3 * 4 * constants.epsilon_0 * constants.c * n0 * np.real(n0)

def chi3_to_Kerr_coeff(chi3, n0, lambda0):
    # derived from the DC Kerr effect formula in https://en.wikipedia.org/wiki/Kerr_effect
    return 3 * chi3 / 2 / n0 / lambda0

def Kerr_coeff_to_chi3(K, n0, lambda0):
    return K / 3 * 2 * n0 * lambda0

def Kerr_coeff_to_delta_n(K, lambda0, E):
    return lambda0 * K * E**2

def chi3_to_delta_n(chi3, n0, lambda0, E):
    K = chi3_to_Kerr_coeff(chi3, n0, lambda0)
    return Kerr_coeff_to_delta_n(K, lambda0, E)

def n2_to_delta_n(n2, n0, lambda0, E):
    chi3 = n2_to_chi3(n2, n0)
    K = chi3_to_Kerr_coeff(chi3, n0, lambda0)
    return Kerr_coeff_to_delta_n(K, lambda0, E)

def r_eff_to_delta_n(r_eff, n0, E):
    return n0**3 / 2 * r_eff * E

def modulation_efficiency_TM(wg, lambda0 = 1.55e-6, dn = 1e-3, plot = False):
    # calculates the modulation efficiency d(n_eff) / d(n_core) of a "wg" object
    # that has attributes d_bcl, d_co, d_tcl (thickness bottom cladding, core, top cladding)
    # and n_cl, n_co (index core and cladding)
    N = 512

    d_stack = [wg.d_bcl, wg.d_co, wg.d_tcl]
    n_stack = [wg.n_cl, wg.n_co, wg.n_cl]
    n_stack_perturbed = [wg.n_cl, wg.n_co + dn, wg.n_cl]

    pos_list, n_list = get_pos_n_list(N, d_stack, n_stack)
    pos_list, n_list_perturbed = get_pos_n_list(N, d_stack, n_stack_perturbed)

    num_modes = 1
    neff_TM, mode_funcs_TM = solve_TM_modes(pos_list, n_list, lambda0, num_modes)
    neff_TM_perturbed, mode_funcs_TM_perturbed = solve_TM_modes(pos_list, n_list_perturbed, lambda0, num_modes)
    dneff_TM = neff_TM_perturbed.real - neff_TM.real
    modulation_efficiency_TM = dneff_TM / dn

    if plot:
        plt.figure(figsize=(8, 2), dpi=100)
        plt.title(f'TM modulation efficiency {modulation_efficiency_TM[0] * 100:.1f}%')
        plt.fill_between(pos_list, 0, np.real(n_list), alpha = 0.2, color = 'k', label = 'index profile')
        plt.twinx()
        plt.plot(pos_list, np.abs(mode_funcs_TM[0]), label = r'TM$_0$')
        plt.xlabel('z (um)')
        plt.legend()

    return modulation_efficiency_TM

########################################################
# Prism coupling angles
########################################################

def index2theta_in_air_to_chip(neff, nprism, nair = 1):
    '''calculates the angle between waveguide tangent and the input beam to
    phase-match the beam with an effective waveguide mode index neff'''
    theta_in_prism_to_chip = np.arccos(neff / nprism)
    theta_in_prism_to_normal = theta_in_prism_to_chip - np.deg2rad(180 - 90 - 45)
    theta_in_air_to_normal = nprism/nair * np.sin(theta_in_prism_to_normal)
    return theta_in_air_to_normal + np.deg2rad(45)

def theta_in_air_to_chip2index(theta_in_air_to_chip, nprism, nair = 1):
    '''calculates the effective waveguide mode index neff from the angle
    between waveguide tangent and the input beam'''
    theta_in_air_to_normal = theta_in_air_to_chip - np.deg2rad(45)
    theta_in_prism_to_normal = np.arcsin(theta_in_air_to_normal * nair/nprism)
    theta_in_prism_to_chip = np.deg2rad(180 - 90 - 45) + theta_in_prism_to_normal
    return nprism * np.cos(theta_in_prism_to_chip)

########################################################
# Fresnel coefficients
########################################################
def r_s(theta_i, n1, n2):
    '''incident and reflected wave in n1,
    transmitted wave in n2
    polarization perpendicular ('senkrecht') to plane of incidence'''
    n = n2/n1
    c = np.sqrt(n**2 - np.sin(theta_i)**2)
    return (np.cos(theta_i) - c ) / (np.cos(theta_i) + c)

def t_s(theta_i, n1, n2):
    '''incident and reflected wave in n1,
    transmitted wave in n2
    polarization perpendicular ('senkrecht') to plane of incidence'''
    n = n2/n1
    c = np.sqrt(n**2 - np.sin(theta_i)**2)
    return 2*np.cos(theta_i) / (np.cos(theta_i) + c)

def r_p(theta_i, n1, n2):
    '''incident and reflected wave in n1,
    transmitted wave in n2
    polarization parallel to plane of incidence'''
    n = n2/n1
    c = np.sqrt(n**2 - np.sin(theta_i)**2)
    return (c - n**2 * np.cos(theta_i)) / (c + n**2 * np.cos(theta_i))

def t_p(theta_i, n1, n2):
    '''incident and reflected wave in n1,
    transmitted wave in n2
    polarization parallel to plane of incidence'''
    n = n2/n1
    c = np.sqrt(n**2 - np.sin(theta_i)**2)
    return 2 * n**2 * np.cos(theta_i) / (c + n**2 * np.cos(theta_i))

def theta_t(theta_i, n1, n2):
    '''incident and reflected wave in n1,
    transmitted wave in n2'''
    return np.arcsin(np.sin(theta_i) * n1 / n2)

def theta_r(theta_i, n1, n2):
    '''incident and reflected wave in n1,
    transmitted wave in n2'''
    return theta_i

########################################################
# TE and TM modes for slab waveguide
#
# ----------------------------
# nc ('cover')
# ---------------------------- 
# nf ('film'), height h
# ----------------------------
# ns ('substrate')
# ----------------------------
#
# k0 - wave vector in air
# kf - transverse wavevector in film
# gamma - attenuation coefficient in cladding
# beta - propagation constant in waveguide
########################################################

def gamma(k0, kf, nf, n):
    return np.sqrt(beta(k0,kf,nf)**2 - k0**2*n**2)

def beta(k0, kf, nf):
    return np.sqrt(k0**2*nf**2 - kf**2)


def rhsTE(kf, k0, nc, nf, ns):
    # left and right hand side of characteristic equation for TE modes
    rhsTE = (gamma(k0, kf, nf, nc) + gamma(k0, kf, nf, ns)) / (kf*(1-gamma(k0, kf, nf, ns)*gamma(k0, kf, nf, nc)/kf**2))
    # Insert a NaN where the difference between successive points is negative/positive
    if type(kf)!=float and type(kf) == np.ndarray:
        rhsTE[:-1][np.diff(rhsTE) > 0] = np.nan
    return rhsTE

def rhsTM(kf, k0, nc, nf, ns):
    # right hand side of TM modes (lhs is the same as TE modes)
    rhsTM = kf*(nf**2/ns**2*gamma(k0, kf, nf, ns) + nf**2/nc**2*gamma(k0, kf, nf, nc)) / (kf**2 - nf**4/ns**2/nc**2*gamma(k0, kf, nf, nc)*gamma(k0, kf, nf, ns))
    if type(kf)!=float and type(kf) == np.ndarray:
        rhsTM[:-1][np.diff(rhsTM) > 0] = np.nan
    return rhsTM

def lhs(kf, h):
    # left hand side of either characteristic equations
    lhs = np.tan(h*kf)
    if type(kf)!=float and type(kf) == np.ndarray:
        lhs[:-1][np.diff(lhs) < 0] = np.nan
    return lhs

def characteristicTE(kf, h, k0, nc, nf, ns):
    return rhsTE(kf, k0, nc, nf, ns) - lhs(kf, h)
    
def characteristicTM(kf, h, k0, nc, nf, ns):
    return rhsTM(kf, k0, nc, nf, ns) - lhs(kf, h)

def kfAllowed_from_sols(sols):
    # filter roots that did not converge
    kfAllowed = []
    for sol in sols:
        if sol[2]==1:
            kfAllowed.append(sol[0])
            
    # filter duplicates
    kfAllowed = np.array(kfAllowed).flatten()
    _, unique = np.unique(kfAllowed.round(decimals=3), return_index=True)
    kfAllowed = kfAllowed[unique]
    
    # filter zeros
    kfAllowed = kfAllowed[np.invert(np.isclose(kfAllowed, 0))]
    
    return kfAllowed

def find_modes(h, k0, nc, nf, ns, analytics = False, n_initial_conditions = 10):
    # returns list of allowed propagation constants in waveguide
    # betaAllowedTE - list of allowed TE propagation constants
    # betaAllowedTE - list of allowed TM propagation constants
    
    # define search space
    kfmax = np.sqrt(k0**2*nf**2 - k0**2*ns**2)
    kf = np.linspace(0, kfmax, 10000)
    # generate equidistant grid of initial conditions of about 10*pi times the number of modes
    V = k0*h*np.sqrt(nf**2 - ns**2)
    x0s = np.linspace(0, kfmax, int(V*n_initial_conditions))

    TEsols = [optimize.fsolve(characteristicTE, x0 = x0, full_output=True, args = (h, k0, nc, nf, ns)) for x0 in x0s]
    TMsols = [optimize.fsolve(characteristicTM, x0 = x0, full_output=True, args = (h, k0, nc, nf, ns)) for x0 in x0s]
    kfAllowedTE = kfAllowed_from_sols(TEsols)
    kfAllowedTM = kfAllowed_from_sols(TMsols)

    betaAllowedTE = beta(k0, kfAllowedTE, nf)
    betaAllowedTM = beta(k0, kfAllowedTM, nf)
    
    
    if analytics:
        print("Beta TE = ", betaAllowedTE)
        print("effective n TE = ", betaAllowedTE/k0)

        print("Beta TM = ", betaAllowedTM)
        print("effective n TM = ", betaAllowedTM/k0)
        
        plt.figure(figsize=(10,3), dpi = 200)
        plt.plot(kf, lhs(kf, h), label = 'LHS', color = 'tab:blue')
        plt.plot(kf, rhsTE(kf, k0, nc, nf, ns), label = "TE polarization", color = 'tab:orange')
        plt.plot(kf, rhsTM(kf, k0, nc, nf, ns), label = "TM polarization", color = 'tab:green')
        plt.legend(loc = "upper left")
        plt.xlabel(r"$\kappa_f$")
        plt.ylim([-30,30])
        for kfTE in kfAllowedTE:
            plt.axvline(kfTE, alpha = 0.5, ls = '--', color = 'tab:orange')
        for kfTM in kfAllowedTM:
            plt.axvline(kfTM, alpha = 0.5, ls = '--', color = 'tab:green')
            
    return betaAllowedTE, betaAllowedTM

### code to get the mode shape
def gamma_from_beta(beta, k0, n):
    return np.sqrt(beta**2 - k0**2*n**2)

def kappa_from_beta(beta, k0, n):
    return np.sqrt(k0**2*n**2 - beta**2)

def cover_mode_shape(x, n, gammac, h, A = 1):
    return A * np.exp(-gammac * (x - h/2))

def film_mode_shape(x, n, kappa, h, A = 1):
    if n%2 == 0:
        return A*np.cos(kappa*x) / np.cos(kappa * h /2)
    if n%2 == 1:
        return A*np.sin(kappa*x) / np.sin(kappa * h /2)
    
def substrate_mode_shape(x, n, gammas, h, A = 1):
    if n%2 == 0:
        return A * np.exp(gammas * (x + h/2))
    if n%2 == 1:
        return -A * np.exp(gammas * (x + h/2))

def get_TE_mode_shape(x, beta, h, k0, ns, nf, nc, mode_number):
    gammas = gamma_from_beta(beta, k0, ns)
    kappa = kappa_from_beta(beta, k0, nf)
    gammac = gamma_from_beta(beta, k0, nc)
    Es = substrate_mode_shape(x[x<-h/2], mode_number, gammas, h)
    Ef = film_mode_shape(x[np.logical_and(x<=h/2, x>=-h/2)], mode_number, kappa, h)
    Ec = cover_mode_shape(x[x>h/2], mode_number, gammac, h)
    return np.concatenate((Es, Ef, Ec))
