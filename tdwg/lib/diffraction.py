from tdwg.lib.ftutils_np import fft_centered_ortho, ifft_centered_ortho, ft_t_axis, ft_f_axis
import numpy as np
from scipy.stats import norm
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()
u.set_enabled_equivalencies(u.dimensionless_angles())
import warnings

def gaussian_beam_from_q_parameter(x, n, k0, q, angle_deg = 0, center = 0):
    input_beam = np.exp(-1j*n*k0*(x-center)**2/(2*q))
    theta = np.deg2rad(angle_deg)
    omega_x = np.tan(theta)*k0*n
    input_beam = input_beam*np.exp(1j*omega_x*x)
    return input_beam

def free_space_propagation(x, cAmps_x, d, k, fresnel_approximation = True):
    kx = 2*np.pi*ft_f_axis(len(x), np.diff(x)[0])
    cAmps_k = fft_centered_ortho(cAmps_x)
    if fresnel_approximation: phase = kx**2 / 2 / k * d # Fresnel phase
    else: phase = -d * np.sqrt(k**2 - kx**2) # Rayleigh-Sommerfeld phase
    cAmps_k *= np.exp(-1j*phase)
    return x, ifft_centered_ortho(cAmps_k)

def focal_plane_to_focal_plane(x, cAmps_x, f, lambda0, clear_aperture = np.inf, *args, **kwargs):
    # if no apertures are involved, simply use the Fourier transform property of a single lens:
    if clear_aperture == np.inf:
        xnew = (ft_f_axis(len(x), np.diff(x)[0])*lambda0*f).to(u.mm)
        dx = np.diff(x)[0]
        dxnew = np.diff(xnew)[0]
        # rescale cAmps in order to preserve the l2 norm (energy)
        cAmps_new = fft_centered_ortho(cAmps_x) * np.sqrt(dx / dxnew)
        # I don't fully understand why I have to do the flip.
        # Physically it is the only thing that makes sense for the beams to propagate in the right direction,
        # yet it disagrees with formula (4.2-8) from Saleh and Teich by a minus sign in the function arguments
        # the flip introduces a tiny offset in the x-axis, which I believe is due to the x-axis not being perfectly centered
        cAmps_new = np.flip(cAmps_new)
        
    # if apertures are involved, use the computationally more expensive propagator below:
    else:
        x, cAmps = free_space_propagation(x, cAmps_x, d = f, k = 2*np.pi/lambda0)
        x, cAmps = parabolic_lens(x, cAmps, f = f, lambda0=lambda0, clear_aperture = clear_aperture, *args, **kwargs)
        xnew, cAmps_new = free_space_propagation(x, cAmps, d = f, k = 2*np.pi/lambda0)
        
    return xnew, cAmps_new

def perfect_4f_setup(x, cAmps, f1, f2, lambda0):
    x, cAmps_k = focal_plane_to_focal_plane(x, cAmps, f1, lambda0)
    x, cAmps = focal_plane_to_focal_plane(x, cAmps_k, f2, lambda0)
    return x, cAmps

def parabolic_lens(x, cAmps_x, f, lambda0, clear_aperture = np.inf, *args, **kwargs):
    # To do: Add warning that triggers when phase is larger than pi/pixel
    cAmps_x = cAmps_x * np.exp(-1j * 2*np.pi/lambda0 * x**2 / 2 / f)
    return aperture(x, cAmps_x, clear_aperture = clear_aperture, *args, **kwargs)

def aperture(x, cAmps_x, clear_aperture = np.inf, power_warning_threshold = 0.01, element_name = 'aperture'):
    mask = (np.abs(x) < clear_aperture / 2).astype(cAmps_x.dtype)
    cAmps_x_after = mask * cAmps_x
    
    if power_warning_threshold is not None:
        power_after = (np.abs(cAmps_x_after)**2).sum()
        power_before = (np.abs(cAmps_x)**2).sum()
        percent_power_lost = (power_before - power_after) / power_before
        if percent_power_lost > power_warning_threshold: 
            print(f"{100*percent_power_lost:.1f}% power was lost at {element_name}, more than {100*power_warning_threshold:.1f}% threshold.")
            
    return x, cAmps_x_after

def focal_plane_to_focal_plane_backwards(x, cAmps_x, f, lambda0, clear_aperture = np.inf):
    if clear_aperture == np.inf:
        cAmps = np.flip(cAmps_x)
        cAmps = ifft_centered_ortho(cAmps)
        x = (ft_f_axis(len(x), np.diff(x)[0])*lambda0*f).to(u.mm)

    else:
        x, cAmps = free_space_propagation_backwards(x, cAmps_x, d = f, k = 2*np.pi/lambda0)
        x, cAmps = parabolic_lens_backwards(x, cAmps, f = f, lambda0=lambda0, clear_aperture = np.inf)
        x, cAmps = free_space_propagation_backwards(x, cAmps, d = f, k = 2*np.pi/lambda0)
        
    return x, cAmps

def perfect_4f_setup_backwards(x, cAmps, f1, f2, lambda0):
    return perfect_4f_setup(x, cAmps, f2, f1, lambda0)

def free_space_propagation_backwards(x, cAmps_x, d, k):
    return free_space_propagation(x, cAmps_x, -d, k)

def parabolic_lens_backwards(x, cAmps_x, f, lambda0, clear_aperture = np.inf):
    return parabolic_lens(x, cAmps_x, -f, lambda0, clear_aperture)