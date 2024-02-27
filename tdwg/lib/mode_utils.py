"""
Just a utility file for simple function that defines modes, which are used in our code
2023-12-25 ported over the functions from "helper_functions.py" -> part of cleanup!
"""
import torch
import astropy.units as u
import numpy as np
from tdwg.lib.beams_utils import get_gaussian

def make_gaussian_modes(x_axis, N, xmode_in_lim, w0, quad_factor=0/u.mm**2):
    """
    2023-12-26: Changed the API a bit, so that it is easier to call from outside!
    """
    x_center_list = np.linspace(-xmode_in_lim, xmode_in_lim, N)
    modes_raw = [torch.from_numpy(get_gaussian(x_axis, x_center, w0, quad_factor*x_center)) for x_center in x_center_list]
    modes_raw = [mode/np.sqrt(torch.sum(torch.abs(mode)**2)) for mode in modes_raw]
    modes = torch.vstack(modes_raw)
    return modes

def make_HG_modes(x_axis, x_center, w0, n):
    modes_raw = [torch.special.hermite_polynomial_h(
        torch.from_numpy(np.sqrt(2)*(x_axis/w0).to('').value), i 
    ) * torch.exp(
        torch.from_numpy(-(x_axis**2 / w0**2).to('').value)
    ) for i in range(n)]
    modes_raw = [mode/np.sqrt(torch.sum(torch.abs(mode)**2)) for mode in modes_raw]
    modes = torch.vstack(modes_raw).to(torch.complex128)
    return modes

def make_boxed_modes(x_axis, N, xmode_out_lim, separation = 0 * u.um):
    x2ind = lambda x: np.argmin(np.abs(x_axis-x)) #make the input less imba, better code
    out_xsep_list = np.linspace(-xmode_out_lim, xmode_out_lim, N+1)
    output_modes = []
    for i in range(N):
        output_mode = np.zeros_like(x_axis).value
        output_mode[x2ind(out_xsep_list[i] + separation/2):x2ind(out_xsep_list[i+1] - separation/2)] = 1
        output_modes.append(output_mode)
    output_modes = torch.from_numpy(np.array(output_modes))
    return output_modes