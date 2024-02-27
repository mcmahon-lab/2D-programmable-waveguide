import numpy as np

A_a10 = (0.25*0.52 / 1.53**2) * 2.54**2 * 1e-4
A_a4 = 0.19 * 0.51 /1.24**2  * 2.54**2 * 1e-4
A_a8 = 0.54 * 0.61 /1.42**2  * 2.54**2 * 1e-4
A_USRNv5 = 0.14 * 0.44 / 1.2**2 * 2.54**2 * 1e-4
A_USRNv3 = 0.55 * 0.14 / 1.23**2 * 2.54**2 * 1e-4
A_a6_1um = 0.44 * 0.31 * (3/3.4)**2 * 2.54**2 * 1e-4
A_a6_2um = 0.4 * 0.48 * (3/3.4)**2 * 2.54**2 * 1e-4
A_a6_3um = 0.4*0.52 * (3/3.4)**2 * 2.54**2 * 1e-4

d_a10 = 470 * 1e-9 
d_a4 = 430 * 1e-9 
d_a8 = 503 * 1e-9
d_USRNv5 = 617 * 1e-9 
d_USRNv3 = 580 * 1e-9
d_a6_1um = 940e-9
d_a6_2um = 2095e-9
d_a6_3um = 2980e-9

eps_r_a10 = 7.40733157
eps_r_a4 = 6.42989993
eps_r_a8 = 7.91684981
eps_r_USRNv5 = 9.34134046
eps_r_USRNv3 = 7.92349142
eps_r_a6_1um = 8.09613761
eps_r_a6_2um = 8.16685487
eps_r_a6_3um = 7.38148588
eps_r_LPCVD_86_5 = 8.866388940124635

def sigma_USRNv3(E, dark):
    if E >= 0:
        if dark:
            return 0.000312*4*140e-9*np.exp(E/1.25e7)
        if not dark:
            return 0.000312/1.55 * 4.0e-3*np.exp(E/3.15e7)
    if E < 0:
        if dark:
            return 0.000312*140e-9*np.exp(-E/0.95e7)
        if not dark:
            return 0.000312 * 4.0e-3*np.exp(-E/3.15e7)
        
def sigma_USRNv5(E, dark):
    if E >= 0:
        if dark:
            return 0.0004998 * 1.7 * 740e-9*np.exp(E/1.3e7)
        if not dark:
            return 0.0004998/5 * 0.9e-2*np.exp(E/1.15e7)
    if E < 0:
        if dark:
            return 0.0004998 * 740e-9*np.exp(-E/0.95e7)
        if not dark:
            return 0.0004998 * 0.9e-2*np.exp(-E/1.65e7)
        
def sigma_a4(E, dark):
    if E >= 0:
        if dark:
            return 0.00011185/4*10e-9*np.exp(E/1.53e7) + 1e-9
        if not dark:
            return 0.00011185/1.1*1.5e-4*np.exp(E/6.05e7) + 1.5e-8
    if E < 0:
        if dark:
            return 0.00011185*10e-9*np.exp(-E/1.53e7) + 1e-9
        if not dark:
            return 0.00011185*1.5e-4*np.exp(-E/4.95e7) + 1.5e-8
        
def sigma_a10(E, dark):
    if E >= 0:
        if dark:
            return 0.00017208/2.1*40e-9*np.exp(E/1.25e7)
        if not dark:
            return 0.00017208*1.05*1.5e-3*np.exp(E/5.95e7)
    if E < 0:
        if dark:
            return 0.00017208*40e-9*np.exp(-E/1.13e7)
        if not dark:
            return 0.00017208*1.5e-3*np.exp(-E/3.95e7)
        
def sigma_a8(E, dark):
    if E >= 0:
        if dark:
            return 0.00039208/2.1*40e-9*np.exp(E/1.45e7)
        if not dark:
            return 0.000312/5.65 * 4.0e-3*np.exp(E/3.15e7)
    if E < 0:
        if dark:
            return 0.00033208*40e-9*np.exp(-E/1.36e7)
        if not dark:
            return 0.00015228*1.5e-3*np.exp(-E/3.05e7)
        
def sigma_a6_1um(E, dark):
    if E >= 0:
        if dark:
            return 2.3e-12*np.exp(E/1.25e7)
        if not dark:
            return 1.75e-7*np.exp(E/5.00e7)
    if E < 0:
        if dark:
            return 6.2e-12*np.exp(-E/1.25e7)
        if not dark:
            return 2.30e-7*np.exp(-E/4.60e7)
            
def sigma_a6_2um(E, dark):
    if E >= 0:
        if dark:
            return 2.7e-12*np.exp(E/1.50e7)
        if not dark:
            return 4.75e-8*np.exp(E/4.00e7)
    if E < 0:
        if dark:
            return 9.7e-12*np.exp(-E/1.50e7)
        if not dark:
            return 1.03e-7*np.exp(-E/4.00e7)
            
def sigma_a6_3um(E, dark):
    if E >= 0:
        if dark:
            return 2e-12*np.exp(E/1.50e7)
        if not dark:
            return 4.6e-8*np.exp(E/4.50e7)
    if E < 0:
        if dark:
            return 7.3e-12*np.exp(-E/1.50e7)
        if not dark:
            return 9.3e-8*np.exp(-E/4.20e7)
        
def sigma_LPCVD_86_5(E, dark):
    if dark:
        return 1.95e-9*np.exp(np.abs(E)/3.50e7)
    if not dark:
        return 2.15e-9*np.exp(np.abs(E)/3.50e7)
        
def exp(x, a, b):
    return a * np.exp(x/b)

def sigma_from_popt(E, popt_dark, popt_bright, popt_dark_negp, popt_bright_negp, dark):
    if E >= 0:
        if dark:
            return exp(E, *popt_dark)
        if not dark:
            return exp(E, *popt_bright)
    if E < 0:
        if dark:
            return exp(-E, *popt_dark_negp)
        if not dark:
            return exp(-E, *popt_bright_negp)
    
data_USRNv5_200C = np.load('tdwg/lib/materials/USRNv5_200C_v2.npz')
popt_dark_USRNv5_200C = data_USRNv5_200C['popt_dark']
popt_bright_USRNv5_200C = data_USRNv5_200C['popt_bright']
popt_dark_negp_USRNv5_200C = data_USRNv5_200C['popt_dark_negp']
popt_bright_negp_USRNv5_200C = data_USRNv5_200C['popt_bright_negp']

def sigma_USRNv5_200C(E, dark):
    return sigma_from_popt(E, 
                           popt_dark_USRNv5_200C, 
                           popt_bright_USRNv5_200C, 
                           popt_dark_negp_USRNv5_200C, 
                           popt_bright_negp_USRNv5_200C,
                           dark)
    
data_SRN10_200C = np.load('tdwg/lib/materials/SRN10_200C_v2.npz')
popt_dark_SRN10_200C = data_SRN10_200C['popt_dark']
popt_bright_SRN10_200C = data_SRN10_200C['popt_bright']
popt_dark_negp_SRN10_200C = data_SRN10_200C['popt_dark_negp']
popt_bright_negp_SRN10_200C = data_SRN10_200C['popt_bright_negp']

def sigma_SRN10_200C(E, dark):
    return sigma_from_popt(E, 
                           popt_dark_SRN10_200C, 
                           popt_bright_SRN10_200C, 
                           popt_dark_negp_SRN10_200C, 
                           popt_bright_negp_SRN10_200C,
                           dark)
    
data_SRN4_200C = np.load('tdwg/lib/materials/SRN4_200C_v2.npz')
popt_dark_SRN4_200C = data_SRN4_200C['popt_dark']
popt_bright_SRN4_200C = data_SRN4_200C['popt_bright']
popt_dark_negp_SRN4_200C = data_SRN4_200C['popt_dark_negp']
popt_bright_negp_SRN4_200C = data_SRN4_200C['popt_bright_negp']

def sigma_SRN4_200C(E, dark):
    return sigma_from_popt(E, 
                           popt_dark_SRN4_200C, 
                           popt_bright_SRN4_200C, 
                           popt_dark_negp_SRN4_200C, 
                           popt_bright_negp_SRN4_200C,
                           dark)