"""
Contain code for reasoning about Gaussian beams. 

Originally, part of this code lived in all different places slaved to either the experimental code or the simulation code. 

Refactoring them here, so that the same code can be used in both places. 

This code assumes the E-field satisfies the following convention
E(z,t) = E_tilde(z, t)*exp(i(kz-wt))
"""

import numpy as np
import astropy.units as u

def get_gaussian(x_in, x_center, w0, fx=0/(1*u.um)):
    """
    Returns a gaussian beam given an x_in axis, the center of the beam, the waist of the beam, and the kx vector of the beam. 
    Inputs
    x_in: the x_axis (most likely the beam at the input plane of the chip), can be from simulation or beamshaper code.
    x_center: the center of the Gaussian beam
    w0: the gaussian beam waist parameters - follows the usual convention (like in wikipedia)
    fx: The x component of the wavevector/(2pi).

    Outputs
    x_in: the x_axis at the 3F plane.
    cAmps_3F: the cAmps at the 3F plane for a gaussian beams
    """
    kx = 2*np.pi*fx
    camps_in = np.exp(-(x_in-x_center)**2/(w0**2))
    camps_in = camps_in*np.exp(1j*(kx*x_in).decompose())
    camps_in = camps_in
    return camps_in

# def get_gaussian(x_in, x_center, w0, k_mat, theta_deg=0):
#     """
#     Returns a gaussian beam that is located at the 3F location. 
#     Inputs
#     x_center: the center of the Gaussian beam
#     w0: the gaussian beam waist parameters - follows the usual convention (like in wikipedia)
#     k_mat: The k vector of the beam IN the material - it is given by n_eff*2*np.pi/lambda
#     theta_deg: the angle that the beam comes in at, by default it is 0

#     Outputs
#     x_in: the x_axis at the 3F plane.
#     cAmps_3F: the cAmps at the 3F plane for a gaussian beams
#     """
#     x_in = self.get_x_in()
#     theta = np.deg2rad(theta_deg)
#     omega_y = np.tan(theta)*k_mat

#     cAmps_3F = np.exp(-(x_in-x_center)**2/(w0**2))
#     cAmps_3F = cAmps_3F*np.exp(1j*omega_y*x_in)
#     cAmps_3F = u.Quantity(cAmps_3F)
#     return x_in, cAmps_3F

def get_q_parameter(z, w0, lambda0, n):
    """
    the minus sign is different from e.g. the wikipedia article on https://en.wikipedia.org/wiki/Complex_beam_parameter,
    since we follow the convention from Goodman's Fourier optics, which places the minus sign in the free-space propagator
    in front of the spatial variable.
    positive z's mean that the beam has propagated past the focal point, negative z's mean that the beam has yet to pass
    the focal point.
    """
    return -z + 1j * np.pi * n * w0**2 / lambda0
