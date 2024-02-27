import numpy as np
import matplotlib.pyplot as plt


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

#############################################################
##### Transfer matrix to also get substrate loss#############
#############################################################
# For the guide on how this mode solver works, see Ryan's thesis. In addition, see the slack post that was written on 2022-02-09 in the 2Dwg channel


def Tinterface(n1, n2, k1, k2):
    """
    n1: index of material at the "top" of interface - see craft notes for schematic
    n2: index of material at the "bottom" of interface - see craft notes for schematic
    """
    T00 = 0.5*(1 + n2**2*k1/(n1**2*k2))
    T01 = 0.5*(1 - n2**2*k1/(n1**2*k2))
    T10 = 0.5*(1 - n2**2*k1/(n1**2*k2))
    T11 = 0.5*(1 + n2**2*k1/(n1**2*k2))

    return np.array([[T00, T01], [T10, T11]])

def Tlayer(ki, Li):
    """
    ki: k vector of layer in the "s" (-y) direction
    Li: length of the layer
    output: returns a transfer matrix that converts the H vector from the top of the layer to the bottom layer
    """
    return np.array([[np.exp(1j*ki*Li), 0], [0, np.exp(-1j*ki*Li)]])

def ki(neff, ni, k0): 
    """
    Ouputs the wavevector in a given layer given the effective index.
    Note that code is way more complex than it needs to be due to the non-continuity of the square root function.
    """
    ki_init = -k0*np.sqrt(ni**2 - neff**2 +0j)
    if np.abs(np.imag(ki_init)) > np.abs(np.real(ki_init)):
        if np.imag(ki_init) > 0:
            ki_ans = -ki_init
        else:
            ki_ans = ki_init
    else: 
        ki_ans = ki_init
    return ki_ans

def transfer_function_loss(neff, L2, L3, k0, n1, n2, n3, nsub):
    """
    The loss function that one needs to minimize in order to find the neff.
    Note here that the imaginary component of neff basically gives the loss of the mode. 
    
    L2 is usually the core height and L3 is the cladding height
    See slack post for a drawing of the schematic
    """
    k1 = ki(neff, n1, k0)
    k2 = ki(neff, n2, k0)
    k3 = ki(neff, n3, k0)
    k4 = ki(neff, nsub, k0)

    a1 = np.array([1, 0])
    T21 = Tinterface(n1, n2, k1, k2)
    T2 = Tlayer(k2, L2)
    T32 = Tinterface(n2, n3, k2, k3)
    T3 = Tlayer(k3, L3)
    T43 = Tinterface(n3, nsub, k3, k4)

    a4 = T43@T3@T32@T2@T21@a1
    return np.abs(a4[0])

def compute_alpha_db_stack(neff_ref, L2, L3, k0, n1, n2, n3, nsub,
                        N_dneff_list = 25, N_kappa_list=50, 
                        flag_visualize=False, vmax=2):
    """
    output: Returns the loss of a stack in units of dB/[distance units that k0 uses]! 
    
    neff_ref: This solver will not work unless you pass a good guess for the effective index. Personally, I would solve for the effective index using find_modes before using this function. 
    
    L2, L3: The height of stack 2 and 3. Both the height of the first and last (substrate layer) is taken to be infinite.
    
    n1, n2, n3: The refractive index of layer 1, 2, and 3 respectively. Of course in practice n1 and n3 are cladding and n2 is the core. NOTE: If you want to add absorption loss into the calculator, make sure that you add a **POSITIVE** imaginary number to the refractive index variable. To double check that this is working, I would also use the flag_visualize variable to check that it is working as expected. 
    
    N_dneff_list, N_kappa_list: number of points used to in sweep to find minimum of the transfer function. 
    
    flag_visualize: Use this when to get a heatmap of the loss function. This is important to do, when you want to visualize that the modesolver is acting as expected. Personally, I would run it with this set to true when checking the solver, and set it to false when a sweep is conducted.
    
    vmax: The vmax that is used in visualization (if it is set to true).
    """
    
    dneff_list = np.logspace(-10, 0, N_dneff_list)
    kappa_list = np.logspace(-10, 0, N_kappa_list)

    loss_mat = np.array([[transfer_function_loss(neff_ref + dn + 1j*kappa, 
                               L2, L3, k0, n1, n2, n3, nsub) 
                          for dn in dneff_list] for kappa in kappa_list])

    ind_min = np.unravel_index(loss_mat.argmin(), loss_mat.shape)
    kappa_opt = kappa_list[ind_min[0]]
    dneff_opt = dneff_list[ind_min[1]]

    alpha = 2*kappa_opt*k0
    alpha_db = alpha*10/np.log(10) #in dB/cm
    
    if flag_visualize:
        plt.figure(figsize=(6, 5), dpi=100)
        plt.pcolormesh(kappa_list, dneff_list, np.log10(loss_mat).T, 
                       cmap="hot", vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label('Log10(Loss)')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Imag(n_eff)")
        plt.ylabel("Real(Δn_eff)")
        plt.axis("square")
        plt.grid(alpha=0.2, linestyle="--")
    
    return alpha_db