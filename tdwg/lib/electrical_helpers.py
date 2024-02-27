import numpy as np
import scipy.constants as const
from scipy.ndimage import convolve1d

def smoothen_current(current, n = 100, mode = 'wrap'):
    kernel = [1/n] * n
    current = convolve1d(current, kernel, axis=- 1, mode=mode)
    return current

def imp_series(Zs):
    return np.sum(np.array(Zs), axis = 0)

def imp_parallel(Zs):
    return 1/np.sum(1/np.array(Zs))

def cap_parallel(Cs):
    return np.sum(np.array(Cs), axis = 0)

def cap_series(Cs):
    return 1/np.sum(1/np.array(Cs))

def imp_cap(f, C):
    return 1/(1j*2*np.pi*f*C)

def parallel_plate_cap(A, d, eps_r = 1):
    return const.epsilon_0 * eps_r * A / d

def resistance2resistivity(R, A, d):
    return R * A / d

def resistivity2resistance(rho, A, d):
    return rho * d / A

def V2E(V, d):
    return V/d

def E2V(E, d):
    return E*d

def waveguide_capacitance_exp(v, i, f):
    Z = v/i
    return 1/2/np.pi/Z/f

def waveguide_capacitance_th(d_cl, d_co, d_cl2, A):
    C_cl = parallel_plate_cap(A, d_cl, eps_r = 3.9)#1.5**2)
    C_co = parallel_plate_cap(A, d_co, eps_r = 3.9*(2.2/1.5)**2)
    C_cl2 = parallel_plate_cap(A, d_cl2, eps_r = 3.9)#1.5**2)
    return cap_series([C_cl, C_co, C_cl2])
    
    
imp_co = lambda f, R : imp_parallel([R, imp_cap(f, C_co)])
imp_cl = lambda f : imp_cap(f, C_cl)
imp_dev = lambda f, R: imp_series([imp_cl(f), imp_co(f, R), imp_cl(f)])

##########################################################
# ----------------------------
# photoconductor (pc)
# ---------------------------- 
# cladding (cl)
# ----------------------------
# core(co)
# ----------------------------
# cladding (cl)
# ----------------------------
#
# C_XX - capacitance of layer XX in Farad
# R_XX - resistance of layer XX in Ohm
# E_XX - electric field across layer XX in V/m
# d_XX - thickness of layer XX in m
##########################################################
from scipy.optimize import fsolve

def voltage_ratios(C_pc, C_cl, C_co, f = None, E_pc = None, R_pc = None):
    # calculates ratio of voltage drop across different layers
    # for a given applied E-field across core
    # E_pc: electric field across core in SI units
    # R_pc: resistance as a function of E_pc of core
    # f: frequency of applied AC voltage
    if R_pc:
        Z_pc = imp_parallel([imp_cap(f, C_pc), R_pc(E_pc)])
    else:
        Z_pc = imp_cap(f, C_pc)
    Z_cl = imp_cap(f, C_cl)
    Z_co = imp_cap(f, C_co)

    Z_total = Z_pc + 2 * Z_cl + Z_co

    r_pc = np.abs(Z_pc / Z_total)
    r_cl = np.abs(Z_cl / Z_total)
    r_co = np.abs(Z_co / Z_total)
    return [r_pc, r_cl, r_co]


def find_E_pc(V, f, R_pc, C_pc, C_cl, C_co, d_pc):
    # returns the roots of the objective function
    # objective function:  
    # Zero if the voltage drop across the core is consistent with the impedance ratio
    f_obj = lambda E_pc: V * voltage_ratios(C_pc, C_cl, C_co, f, E_pc, R_pc)[0] - E_pc * d_pc

    # initial guess: all voltage drops across the core
    E_pc = fsolve(f_obj, V/d_pc)
    
    return E_pc

def Z_total(f, R_pc, C_pc, C_cl, C_co):
    # returns absolute value of total impedance
    Z_pc = imp_parallel([imp_cap(f, C_pc), R_pc])
    Z_cl = imp_cap(f, C_cl)
    Z_co = imp_cap(f, C_co)

    Z_total = Z_pc + 2 * Z_cl + Z_co
    return np.abs(Z_total)