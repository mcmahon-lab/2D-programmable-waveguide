"""
Some notes on notation. 

x_bs, camps_in: These are the fields that are going to be instantiated right at the input facet of the chip - I have decided to denote 
x_3F, camps_3F: These are the fields that are the front focal plane of the lens L3.

Now phenomelogically, camps_in and camps_3F will be mapped between each other with some element-wise complex number multiplication.
Note that x_bs = x_3F in the current scheme. To make things clear, I will be switching from x_3F to x_bs (x_3F was used previously.)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from ctypes import *
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import norm
import collections
import astropy.units as u
import time

from tdwg.lib.ftutils_np import ft_f_axis
from tdwg.lib.plot_utils import plot_norm
from tdwg.lib import diffraction

eps = 1e-4 #you need this to get rid of the floating point comparison!

class Beamshaper():
    """
    Massive change in the code, I want to use this as a playground to make sure that the code that I write is equivalent...
    """
    def __init__(self, calibration_dict, period_width = 16, online_flag = True):   
        self.pixelPitch = 8 * u.um
        self.periodWidth = 16 #hardcode for now
        self.load_calibration(calibration_dict)
        self.online_flag = online_flag

        if self.online_flag:
            from tdwg.lib import slmpy
            self.slm = slmpy.SLMdisplay(monitor = 1)
            self.resLong, self.resShort = self.slm.getSize()
        else:
            self.resLong = 1920
            self.resShort = 1200
        
        self.longAxisPx  = (np.arange(self.resLong) - self.resLong/2)
        self.longAxis = self.longAxisPx * self.pixelPitch.to(u.mm)
        self.shortAxisPx  = (np.arange(self.resShort) - self.resShort/2)
        self.shortAxis = self.shortAxisPx * self.pixelPitch.to(u.mm)

        self.lambda0 = 1.55*u.um
        self.k = 2*np.pi / self.lambda0
        
        self.f1 = 500*u.mm
        self.f2 = 200*u.mm
        self.f3 = 4.5*u.mm
        
        self.aperture_L1 = 2*u.imperial.inch
        self.aperture_L2 = 22*u.mm
        self.aperture_L3 = 26.5*u.mm
        
        self.delta_1_SLM = -9.97698960*u.mm # genetic algorithm fit from 23-02-08
        
        self.x_high_res = np.linspace(-(self.f1 * self.lambda0 / self.pixelPitch/2).to('mm').value, 
                                      (self.f1 * self.lambda0 / self.pixelPitch/2).to('mm').value, 
                                      60000)*u.mm
        
        self.load_magnitude_calibration()

    def load_calibration(self, calibration_dict):
        self.angle = calibration_dict['beamshaper_angle']
        self.phase_quad = calibration_dict['phase_quad']
        self.phase_cubic = calibration_dict['phase_cubic']
        self.phase_linear = calibration_dict['phase_linear'] #this is the shift in the frequency domain, to align with DMD

    def get_img(self, mags, phases):
        #The flipping is necessary because the SLM is flipped.
        mags = np.flip(mags)
        phases = np.flip(phases)

        #Ok, see if I can fix it here... This is new code from 04-19
        phases = -phases #this will do a conjugation
        #

        inds = np.arange(self.resLong)
        grating = sawtooth(np.add.outer(inds*2*np.pi/self.periodWidth, -phases) +eps, width = 0)
        grating -= 1/self.periodWidth
        grating *= mags/2
        grating += 0.5
        grating *= 255
        return grating.T.astype('uint8')

    def rotateSLMImg(self, img, angle):
        img_rotated = rotate(img, angle)
        shape_diff = np.abs(np.array(img.shape) - np.array(img_rotated.shape))
        img_rotated = img_rotated[int(shape_diff[0]/2):int(shape_diff[0]/2) + self.resShort,
                                int(shape_diff[1]/2):int(shape_diff[1]/2) + self.resLong,]
        return img_rotated
    
    def apply_mags_phases(self, mags, phases, linearize = True):
        """
        Linearity correction is applied before applying shape to SLM.
        Also rotates the image by self.angle
        Processing is done so as to be consistent with conventions.

        Inputs:
        mags: Magnitudes of the complex field. (s) stands for a list of magnitude. 
        phases: Phases of the complex field. 
        
        """
        if any(mags > 1+1e-10):
            print('One of the SLM magnitude is larger than 1! Check normalization.')
        if linearize:
            mags = self.linearize_mags(mags)
        img = self.get_img(mags, phases)
        img_rotated = self.rotateSLMImg(img, self.angle)
        if self.online_flag:
            self.slm.updateArray(img_rotated)
        self.img_applied = img_rotated #img_applied refers to the final image that is applied to the SLM.
        
    def apply(self, camps, linearize = True):
        """
        camps: Complex amplitudes of the field.
        """
        mags = np.abs(camps)
        phases = np.angle(camps)
        self.apply_mags_phases(mags, phases, linearize = linearize)

    def linearize_mags(self, mags):
        # find magnitudes to apply to the SLM such that the power
        # in the first order diffraction maximum is proportional to "mags"
        diff = np.abs(np.array(list(self.inv_mag_LUT.keys())) - np.array([mags]).T)
        inds = np.argmin(diff, axis = -1)
        return np.array(list(self.inv_mag_LUT.values()))[inds]
    
    def phase_warning(self, mags, phases):
        # an indicator function that can be used to judge whether a particular output
        # requires a too high frequency phase pattern on the SLM. 
        # the closer to zero, the better, the closer to 1, the worse
        mags_norm = mags / mags.max()
        phases_unwrapped = np.unwrap(phases).value
        indicator = np.diff(phases_unwrapped) * mags_norm[1:] / np.pi
        return np.abs(indicator).max()

    def camps_3F_to_camps_SLM(self, x_3F, camps_3F, normalize = True):
        """
        Need to rewrite the documentation of this function more carefully...

        This will be quite an imba function, in that it will take the intended complex field at the 
        3F plane (aka where the chip is), and return the camps that needs to be applied to the SLM

        It also takes care of all of the renormalization and flipping, so the user does not need to do that.
        I decided to not write another private function that doesn't do the renormalization
        as I figured that there is almost no application where the raw function is helpful to be accessible.
        But of course, if the time arrise, the refactoring of this code may be warranted.
        """
        x_2F, camps_2F = diffraction.focal_plane_to_focal_plane_backwards(
            x_3F, camps_3F, f = self.f3, lambda0 = self.lambda0)
        x_1F, camps_1F = diffraction.focal_plane_to_focal_plane_backwards(
            x_2F, camps_2F, f = self.f2, lambda0 = self.lambda0)
        x_1B, camps_1B = diffraction.focal_plane_to_focal_plane_backwards(
            x_1F, camps_1F, f = self.f1, lambda0 = self.lambda0)
        
        x_SLM, camps_SLM = x_1B, camps_1B
        # 2023-04-19 comment this out, so that I can do this in an outer function
        # x_SLM, camps_SLM = diffraction.free_space_propagation_backwards(
            # x_1B, camps_1B, d = self.delta_1_SLM, k = self.k)

        if normalize:
            camps_SLM /= (np.abs(camps_SLM).max() )

        return x_SLM, camps_SLM

    def get_x_bs(self):
        """
        Returns the x_bs vector that is usable for the construction of input fields!
        Must use this if want to have success with using the all-important camps_3F_to_camps_SLM function
        """
        x_SLM = self.shortAxis
        xmax = (self.lambda0 * self.f1 / np.diff(x_SLM)[0]).to(u.mm)
        xnew = (ft_f_axis(len(x_SLM), np.diff(x_SLM)[0])*self.lambda0*self.f1).to(u.mm)
        x_bs = xnew * self.f3 / self.f2
        self.x_bs = x_bs
        return x_bs

    def apply_Ein(self, Ein_x_bs, sleep_time):
        x_bs = self.x_bs
        phase_quad = self.phase_quad
        phase_cubic = self.phase_cubic
        phase_linear = self.phase_linear

        calibration_vector = np.exp(1j*2*np.pi*(phase_cubic*x_bs**3/3 + phase_quad*x_bs**2/2 + phase_linear*x_bs))
        camps_3F = Ein_x_bs*calibration_vector
        _, Ein_x_SLM = self.camps_3F_to_camps_SLM(x_bs, camps_3F, normalize=True)
        self.apply(Ein_x_SLM)
        time.sleep(sleep_time)

    def load_magnitude_calibration(self, plot = False):
        """
        load calibration data for grating amplitude to diffraction efficiency and amplitude profile of beam illuminating the SLM

        Hiro: Note that I haven't had time to rewrite this function to be consistent with the conventions of using "mag", which stands for magnitude, in this file. Instead, the term "amp" is used, which stands for amplitude/magnitude.
        """
        data = np.load('tdwg/lib/beamshaper_calibration.npz')

        amp_list = data['amp_list']
        loc_list = data['loc_list']
        power_list = data['power_list']
        power_list /= power_list.max()
        ind_maxpower = np.argmax(power_list[:,-1])

        # create finer amplitude list and store interpolated values in dictionary
        amp_list_fine = np.linspace(0,1, 1000)
        spl = UnivariateSpline(amp_list, np.sqrt(power_list[ind_maxpower]), s = 0)
        amplitudeLUT = collections.OrderedDict(zip(amp_list_fine, spl(amp_list_fine)))
        # invert dictionary
        self.inv_mag_LUT = collections.OrderedDict({v: k for k, v in amplitudeLUT.items()})

        if plot:
            plt.title('Amplitude calibration')
            plot_norm(amp_list, np.sqrt(power_list[ind_maxpower]), '.--', label = 'calibration curve')
            plot_norm(amp_list_fine, spl(amp_list_fine), alpha = 0.5, label = 'fit')
            plt.xlabel('Applied amplitude')
            plt.ylabel('Measured amplitude')
            plt.legend()
            plt.show()
            
        amp_loc = np.sqrt(power_list[:, -1])
        def gaussian_amp1(x, loc, scale):
            gaussian = norm.pdf(x, loc = loc, scale = scale)
            return gaussian/gaussian.max()
        popt, pcov = curve_fit(gaussian_amp1, loc_list, amp_loc, p0 = [0, 1])

        self.beam_profile = gaussian_amp1(self.shortAxis, *popt)

        if plot:
            plt.plot(loc_list, np.sqrt(power_list[:, -1]), '.--', label = 'calibration curve')
            plt.plot(self.shortAxis, self.beam_profile, label = 'fit')
            plt.title('Beam profile')
            plt.xlabel('Location on SLM')
            plt.ylabel('Measured amplitude')
            plt.legend()
            plt.show()

def pos2ind(axis, x):
    return np.argmin(np.abs(axis-x))