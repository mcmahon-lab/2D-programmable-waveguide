import numpy as np
import matplotlib.pyplot as plt

def get_pos_n_list(N, d_stack, n_stack):
    L_total = np.sum(d_stack)
    pos_list = np.linspace(0, L_total, N)

    pos_cuts = [0.0] + list(np.cumsum(d_stack))
    n_list = np.zeros_like(pos_list)*1j #to make it complex
    pos2ind = lambda pos: np.argmin((pos_list-pos)**2)

    for i in range(len(pos_cuts)-1):
        n_list[pos2ind(pos_cuts[i]):pos2ind(pos_cuts[i+1])] = n_stack[i]
    n_list[-1] = n_stack[-1] #an edge case that I need to take care of because of how numpy indexing works
    return pos_list, n_list

from scipy.linalg import circulant
from scipy.sparse.linalg import eigen

def solve_TE_modes(pos_list, n_list, λ, num_modes):
    """
    Inputs:
    pos_list: vector of positions for n_list - assumed to be equally spaced!
    n_list: vector of refractive index at each point in pos_list - can be complex if loss is desired
    λ: wavelength of operation, in same units as pos_list - recommend um
    num_eigs: number of modes that will be solved for
    
    Outputs:
    neff: list of neff, output is complex so if loss is desired take the imag part
    mode_funcs: The TE mode functions that are found.
                Note that I normalize them such that np.sum(np.abs(mode_funcs[i])**2)*dy = 1
                The ideal is that the output is the mode function sampled at pos_list locations
    """
    dy = np.diff(pos_list)[0]#to have the discretization
    N = len(pos_list)

    kernel = [-2, 1] + [0]*(N-3) + [1]
    second_deriv = circulant(kernel)/dy**2
    second_deriv[-1, 0] = 0 #so as to not have circular boundary conditions, honestly probably doesn't matter if you have the two lines below
    second_deriv[0, -1] = 0

    k0 = 2*np.pi/λ
    M = second_deriv + k0**2*np.diag(n_list)**2
    eigenval, eigenvec = eigen.eigs(M, k=num_modes, which="LR")

    mode_funcs = eigenvec.T/np.sqrt(dy)
    neff = np.sqrt(eigenval/k0**2)
    return neff, mode_funcs

from scipy.interpolate import interp1d

def solve_TM_modes(pos_list, n_list, λ, num_modes):
    """
    Inputs:
    pos_list: vector of positions for n_list - assumed to be equally spaced!
    n_list: vector of refractive index at each point in pos_list - can be complex if loss is desired
    λ: wavelength of operation, in same units as pos_list - recommend um
    num_eigs: number of modes that will be solved for
    
    Outputs:
    neff: list of neff, output is complex so if loss is desired take the imag part
    mode_funcs: The TM mode functions that are found.
                Note that I normalize them such that np.sum(np.abs(mode_funcs[i])**2)*dy = 1
                The ideal is that the output is the mode function sampled at pos_list locations
    """
    dy = np.diff(pos_list)[0]#to have the discretization
    N = len(pos_list)

    #Here subsample the refractive into a grid that is twice as fine. Require this, see derivation in slack post
    pos_list_fine = np.linspace(pos_list[0], pos_list[-1] + dy/2, 2*N)
    interp_func = interp1d(pos_list, n_list, fill_value="extrapolate")
    n_fine = interp_func(pos_list_fine) #it should be called n_list_fine but calling it n_fine for brievity

    A = np.zeros([N, N])*0j

    for i in np.arange(1, N-1):
        A[i, i+1] = n_fine[2*(i+1)]**2/n_fine[2*i+1]**2
        A[i, i] = -n_fine[2*i]**2*(1/n_fine[2*i+1]**2 + 1/n_fine[2*i-1]**2)
        A[i, i-1] = n_fine[2*(i-1)]**2/n_fine[2*i-1]**2

    A = A/(dy**2)

    k0 = 2*np.pi/λ
    M = A + k0**2*np.diag(n_list)**2
    eigenval, eigenvec = eigen.eigs(M, k=num_modes, which="LR")

    mode_funcs = eigenvec.T/np.sqrt(dy)
    neff = np.sqrt(eigenval/k0**2)
    return neff, mode_funcs