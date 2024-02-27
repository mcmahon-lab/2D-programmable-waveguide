"""
2D-diffraction package
unlike the 1D-diffraction package (diffraction.py), this package completely lacks finite apertures
"""

from tdwg.lib.ftutils_np import fft2_centered_ortho, ifft2_centered_ortho, ft_t_axis, ft_f_axis
import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()
u.set_enabled_equivalencies(u.dimensionless_angles())
from tdwg.lib.ftutils_np import fft2_centered_ortho
import warnings

def gaussian_beam_from_q_parameter(xx, yy, n, k0, q, angle_x_deg = 0, angle_y_deg = 0, center_x = 0, center_y = 0):
    input_beam = np.exp(-1j*n*k0*((xx-center_x)**2 + (yy-center_y)**2)/(2*q))
    theta_x = np.deg2rad(angle_x_deg)
    theta_y = np.deg2rad(angle_y_deg)
    omega_x = np.tan(theta_x)*k0*n
    omega_y = np.tan(theta_y)*k0*n
    input_beam = input_beam*np.exp(1j*omega_x*xx)*np.exp(1j*omega_x*yy)
    return input_beam

def free_space_propagation(xx, yy, cAmps_xy, d, k, fresnel_approximation = True):
    x = xx[0]
    y = yy[:,0]
    kx = 2*np.pi*ft_f_axis(len(x), np.diff(x)[0])
    ky = 2*np.pi*ft_f_axis(len(y), np.diff(y)[0])
    kxx, kyy = np.meshgrid(kx, ky)
    cAmps_k = fft2_centered_ortho(cAmps_xy)
    if fresnel_approximation: phase = (kxx**2 + kyy**2) / 2 / k * d # Fresnel phase
    else: phase = -d * np.sqrt(k**2 - kxx**2 - kyy**2) # Rayleigh-Sommerfeld phase
    cAmps_k *= np.exp(-1j*phase)
    return xx, yy, ifft2_centered_ortho(cAmps_k)

def aperture_mask(xx, yy, center_x = 0, center_y = 0, clear_aperture = np.inf):
    mask = (np.sqrt((xx-center_x)**2 + (yy-center_y)**2) < clear_aperture / 2)
    return mask

def aperture(xx, yy, cAmps_xy, center_x = 0, center_y = 0, clear_aperture = np.inf, power_warning_threshold = None, element_name = 'aperture'):
    mask = aperture_mask(xx, yy, center_x, center_y, clear_aperture).astype(cAmps_xy.dtype)
    cAmps_xy_after = mask * cAmps_xy
    
    if power_warning_threshold is not None:
        power_after = (np.abs(cAmps_xy_after)**2).sum()
        power_before = (np.abs(cAmps_xy)**2).sum()
        percent_power_lost = (power_before - power_after) / power_before
        if percent_power_lost > power_warning_threshold: 
            print(f"{100*percent_power_lost:.1f}% power was lost at {element_name}, more than {100*power_warning_threshold:.1f}% threshold.")
            
    return xx, yy, cAmps_xy_after

def phase_warning(xx, yy, phase, center_x, center_y, clear_aperture):
    """
    prints a warning if the variable phase changes faster than pi per pixel 
    anywhere within the clear aperture diameter around (center_x, center_y)
    """
    if clear_aperture == np.inf: mask = 1
    else: mask = aperture_mask(xx, yy, center_x, center_y, clear_aperture)
    masked_phase = mask*phase
    phase_change_magnitude_x = np.abs(np.diff(masked_phase, axis = 0))
    phase_change_magnitude_y = np.abs(np.diff(masked_phase, axis = 1))
    mask_bound_x = np.diff(mask.data, axis = 0)
    mask_bound_y = np.diff(mask.data, axis = 1)
    max_phase_change_x = np.max(phase_change_magnitude_x[np.logical_not(mask_bound_x)])
    max_phase_change_y = np.max(phase_change_magnitude_y[np.logical_not(mask_bound_y)])
    if max_phase_change_x > np.pi or  max_phase_change_y > np.pi:
        print(f"The phase changes too rapidly at the lens. (Maximal phase change per pixel = {np.max((max_phase_change_x, max_phase_change_y)):.1f}>pi). This might result in numerical artifacts. \n\
        Either: \n\
        1) Decrease pixel size, \n\
        2) Decrease simulation area or lens aperture, \n\
        3) increase lens focal length")
        

def parabolic_lens(xx, yy, cAmps_xy, f, k, center_x = 0, center_y = 0, clear_aperture = np.inf, *args, **kwargs):
    phase = k * ((xx-center_x)**2 + (yy-center_y)**2) / 2 / f
    cAmps_xy = cAmps_xy * np.exp(-1j * phase)
    phase_warning(xx, yy, phase, center_x, center_y, clear_aperture)
    return aperture(xx, yy, cAmps_xy, center_x, center_y, clear_aperture, *args, **kwargs)

def distance_from_angled_line(xx, yy, theta_deg, center_x = 0, center_y = 0):
    xx = xx - center_x
    yy = yy - center_y
    theta = np.deg2rad(theta_deg)
    return np.sqrt((xx - (xx * np.cos(theta) + yy * np.sin(theta)) * np.cos(theta))**2 + (yy - (xx * np.cos(theta) + yy * np.sin(theta)) * np.sin(theta))**2)

def cylindrical_lens(xx, yy, cAmps_xy, f, k, angle_deg=0, center_x = 0, center_y = 0, clear_aperture = np.inf, *args, **kwargs):
    """
    applies the parabolix phase shift of a cylindrical lens with angle = 0 meaning the 
    cylindrical lens lies flat in the x-axis (i.e. a gaussian beam will form a 
    line parallel with the x-axis after passing through the lens)
    """
    d = distance_from_angled_line(xx, yy, angle_deg, center_x = center_x, center_y=center_y)
    phase = k * d**2 / 2 / f
    cAmps_xy = cAmps_xy * np.exp(-1j * phase)
    phase_warning(xx, yy, phase, center_x, center_y, clear_aperture)
    return aperture(xx, yy, cAmps_xy, center_x, center_y, clear_aperture, *args, **kwargs)

def focal_plane_to_focal_plane(xx, yy, cAmps_xy, f, lambda0):
    x = xx[0]
    y = yy[:,0]
    xnew = ft_f_axis(len(x), np.diff(x)[0])*lambda0*f
    ynew = ft_f_axis(len(y), np.diff(y)[0])*lambda0*f
    xx, yy = np.meshgrid(xnew, ynew)
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    dxnew = np.diff(xnew)[0]
    dynew = np.diff(ynew)[0]
    # rescale cAmps in order to preserve the l2 norm (energy)
    cAmps_new = fft2_centered_ortho(cAmps_xy) * np.sqrt(dx / dxnew * dy / dynew)
        
    return xx, yy, cAmps_new

def perfect_4f_setup(xx, yy, cAmps, f1, f2, lambda0):
    xx, yy, cAmps_k = focal_plane_to_focal_plane(xx, yy, cAmps, f1, lambda0)
    xx, yy, cAmps = focal_plane_to_focal_plane(xx, yy, cAmps_k, f2, lambda0)
    return xx, yy, cAmps

def power(xx, yy, cAmps_xy):
    x = xx[0]
    y = yy[:,0]
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    return np.sum(dx*dy*np.abs(cAmps_xy)**2)