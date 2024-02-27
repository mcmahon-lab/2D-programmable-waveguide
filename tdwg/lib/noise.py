import numpy as np
import matplotlib.pyplot as plt
from tdwg.lib.ftutils_np import *
import warnings

def make_pink_noise(x_axis, y_axis, noise_rms, noise_len_min, noise_len_max, skew=1, seed = None):
    """
    Inputs: 
    x_axis: x_axis of the boxed noise, must evenly spaced!
    y_axis: y_axis of the boxed noise, must evenly spaced!
    skew: Represents the stretching of the axis. skew > 1 means that the noise will be stretched out in the x-dimension
    noise_amp represents the variance of the states
    """
    if noise_len_max == None:
        f_min = 0
    else:
        f_min = 1/noise_len_max

    f_max = 1/noise_len_min

    Nx, Ny = len(x_axis), len(y_axis)
    dx = np.diff(x_axis)[0]
    dy = np.diff(y_axis)[0]

    fx_axis = ft_f_axis(Nx, dx)
    fy_axis = ft_f_axis(Ny, dy)
    fx_mesh, fy_mesh = np.meshgrid(fx_axis, fy_axis, indexing='ij')

    dfx = np.diff(fx_axis)[0]
    dfy = np.diff(fy_axis)[0]

    if seed is not None:
        np.random.seed(seed)
    # noise_f = 1/np.sqrt(2)*(np.random.randn(Nx, Ny) + 1j*np.random.randn(Nx,Ny))
    dist_center = np.sqrt(fx_mesh**2*skew**2 + fy_mesh**2)
    noise_f = 1*(np.random.randn(Nx, Ny) + 1j*np.random.randn(Nx,Ny))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_mod = np.ones([Nx, Ny])/dist_center
    noise_mod[dist_center > f_max] = 0.0
    noise_mod[dist_center < f_min] = 0.0

    total_energy = np.sum(np.abs(noise_mod)**2)*dfx*dfy

    Lx = x_axis[-1]-x_axis[0]
    Ly = y_axis[-1]-y_axis[0]

    noise_mod_factor = np.sqrt(total_energy/(Lx*Ly))

    noise_f = noise_rms*np.array(noise_mod)*noise_f/noise_mod_factor
    noise_xy = ifft2_centered_iso(noise_f, dx, dy)

    return noise_xy, noise_f

def make_boxed_noise(x_axis, y_axis, noise_amp, noise_len_min, noise_len_max=None, seed = None):
    """
    Inputs: 
    x_axis: x_axis of the boxed noise, must evenly spaced!
    y_axis: y_axis of the boxed noise, must evenly spaced!
    """
    if noise_len_max == None:
        f_min = 0
    else:
        f_min = 1/noise_len_max
        
    f_max = 1/noise_len_min
    
    Nx, Ny = len(x_axis), len(y_axis)
    dx = np.diff(x_axis)[0]
    dy = np.diff(y_axis)[0]
    
    fx_axis = ft_f_axis(Nx, dx)
    fy_axis = ft_f_axis(Ny, dy)
    fx_mesh, fy_mesh = np.meshgrid(fx_axis, fy_axis, indexing='ij')
    
    dfx = np.diff(fx_axis)[0]
    dfy = np.diff(fy_axis)[0]
    
    
    if seed: np.random.seed(seed)
    # noise_f = 1/np.sqrt(2)*(np.random.randn(Nx, Ny) + 1j*np.random.randn(Nx,Ny))
    noise_f = 1*(np.random.randn(Nx, Ny) + 1j*np.random.randn(Nx,Ny))

    dist_center = np.sqrt(fx_mesh**2 + fy_mesh**2)
    noise_mod = np.ones([Nx, Ny])
    noise_mod[dist_center > f_max] = 0.0
    noise_mod[dist_center < f_min] = 0.0

    #the amplitude can be arrived at analytically
    factor = 1/np.sqrt(np.pi*(f_max**2-f_min**2)*dfx*dfy)
    
    noise_f = factor*noise_amp*np.array(noise_mod)*noise_f
    noise_xy = ifft2_centered_iso(noise_f, dx, dy)
    
    return noise_xy, noise_f

def plot_f_noise(fx_axis, fy_axis, noise_f):
    plt.pcolormesh(fx_axis, fy_axis, np.log(np.abs(noise_f)).T, cmap="binary")
    plt.colorbar()
    plt.axis("equal")

def plot_xy_noise(x_axis, y_axis, noise_xy):
    #to see the pink nature of the noise, need to parse out
    #a limited portion of the initial noise
    x2ind = lambda x: np.argmin((x_axis-x)**2)
    y2ind = lambda y: np.argmin((y_axis-y)**2)
    
    Nx, Ny = noise_xy.shape
    
    x_axis_lim = x_axis[:int(Nx/10)]
    y_axis_lim = y_axis[:int(Ny/10)]
    noise_xy_lim = noise_xy[:int(Nx/10), :int(Ny/10)]

    fig, axs = plt.subplots(2, 1, figsize=(6,6), dpi=100)

    plt.sca(axs[0])
    plt.pcolormesh(x_axis, y_axis, np.real(noise_xy).T, cmap="binary")
    plt.axis("equal")
    plt.colorbar()

    plt.sca(axs[1])
    plt.pcolormesh(x_axis_lim, y_axis_lim, np.real(noise_xy_lim).T, cmap="binary")
    plt.axis("equal")
    plt.colorbar()

