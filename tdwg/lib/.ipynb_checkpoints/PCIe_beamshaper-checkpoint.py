import os
import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import norm
import collections
import astropy.units as u
import time
import copy
from skimage.util.shape import view_as_windows as viewW
import warnings

from tdwg.lib.DMD_patterns import generate_all_on
from tdwg.lib.ftutils_np import ft_f_axis
from tdwg.lib.plot_utils import plot_norm
from tdwg.lib import diffraction
from tdwg.lib.sweeps import sweep_zip

from tdwg.lib.multiuser_client import Client
client = Client()

eps = 1e-4 #you need this to get rid of the floating point comparison!

########## Hardcoded parameters of SLM #############
#I decided to hardcode these! I think it does make more sense in the long term anyways, since realistically, one needs to edit this file when using a different SLM anyways! Also it makes the code much much cleaner...
resLong = 1024
resShort = 1024
pixelPitch = 17 * u.um

# Simple derived quantities
shortAxisPx  = (np.arange(resShort) - resShort/2)
shortAxis = shortAxisPx * pixelPitch.to(u.mm)


################ Set of code for generating the grating! ##################
def pos2ind(axis, x):
    return np.argmin(np.abs(axis-x))

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]
    
def fast_modulo_256(array_uint16):
    """Takes uint16 array and returns (array_uint16 % 256) as uint8 array"""
    return array_uint16.view('uint8')[:,0::2]

def fast_division_256(array_uint16):
    """Takes uint16 array and returns (array_uint16 / 256) as uint8 array"""
    return array_uint16.view('uint8')[:,1::2]

def get_rotated_img(mags, phases, periodWidth, angle):
    """
    creates rotated 2D grating to be displayed on beamshaper with columns
    corresponding to magnitudes "mags" and phases "phases"
    
    A pure function! So it can be defined here.
    
    angle is in degrees!
    """
    pixelValues8 = 256
    # spacing between values of adjacent pixels in column
    pixelsPerPeriod8 = int(pixelValues8/periodWidth)

    mags = np.flip(mags)
    phases = np.flip(phases)

    #Ok, see if I can fix it here... This is new code from 04-19
    phases = -phases #this will do a conjugation

    phases = phases.to('rad').value/2/np.pi*pixelValues8
    phases = phases.astype('uint16')

    # create grating of appropriate period
    inds = np.arange(resLong).astype('uint16')
    pregrating = np.add.outer(inds*pixelsPerPeriod8, -phases)
    grating = fast_modulo_256(pregrating).astype('int8')

    # shift grating from 0->256 to -128->128
    grating -= int(pixelValues8/2)
    grating += int(pixelsPerPeriod8/2)

    # scale grating to correct magnitude and shift back to positive values only
    # this could be sped up further by performing the multiplication as 
    # int16 rather than float.
    grating = (grating*mags).astype('int8')
    grating = (grating+127).astype('uint8')

    # rotate grating by sliding different rows by different amounts
    rotation_px_shift = np.linspace(-512*np.deg2rad(angle), 512*np.deg2rad(angle), 1024).astype('int8')
    rotated_grating = strided_indexing_roll(grating, rotation_px_shift)
    return np.transpose(rotated_grating).copy().astype('uint8')


######### Classes related to the beamshaper!!!!!!!#############

class PCIe_SLM():
    def __init__(self):
        # Load the DLL
        # Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
        # should all be located in the same directory as the program referencing the
        # library
        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")

        # Open the image generation library
        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\ImageGen")
        image_lib = CDLL("ImageGen")

        # Basic parameters for calling Create_SDK
        # these parameters are further explained on page 24 onwards in th "PCIe User Manual.pdf"
        bit_depth = c_uint(12)
        num_boards_found = c_uint(0)
        constructed_okay = c_uint(-1)
        is_nematic_type = c_bool(1)
        RAM_write_enable = c_bool(1)
        use_GPU = c_bool(1)
        max_transients = c_uint(20)
        self.board_number = c_uint(1)
        self.wait_For_Trigger = c_uint(0)
        self.flip_immediate = c_uint(0) #only supported on the 1024
        self.timeout_ms = c_uint(5000)
        center_x = c_float(256)
        center_y = c_float(256)
        VortexCharge = c_uint(3)
        fork = c_uint(0)
        RGB = c_uint(0)

        # Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
        # or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
        self.OutputPulseImageFlip = c_uint(0)
        self.OutputPulseImageRefresh = c_uint(0); #only supported on 1920x1152, FW rev 1.8. 

        # Call the Create_SDK constructor
        # Returns a handle that's passed to subsequent SDK calls
        self.slm_lib.Create_SDK(bit_depth, byref(num_boards_found), byref(constructed_okay), is_nematic_type, RAM_write_enable, use_GPU, max_transients, 0)

        if constructed_okay.value == 0:
            print ("Blink SDK did not construct successfully");
        else:
            print ("Blink SDK was successfully constructed");
            
        print ("Found %s SLM controller(s)" % num_boards_found.value)
        depth = c_uint(self.slm_lib.Get_image_depth(self.board_number)); #Bits per pixel
        Bytes = c_uint(depth.value//8);
        center_x = c_uint(resShort//2);
        center_y = c_uint(resLong//2);
        
        # load lookup table
        self.slm_lib.Load_LUT_file(self.board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm20230822_40C_at1550.lut")
        
        # wavefront correction
        WFC = np.zeros([resShort*resLong*Bytes.value], np.uint8, 'C');
        WFC = WFC.ctypes.data_as(POINTER(c_ubyte))
        
    def updateArray(self, img):
        img = img.astype('uint8')
        self.applied_img = img
        img = img.ctypes.data_as(POINTER(c_ubyte))
        self.slm_lib.Write_image(self.board_number, img, resLong*resShort, self.wait_For_Trigger, self.flip_immediate, self.OutputPulseImageFlip, self.OutputPulseImageRefresh, self.timeout_ms)
        
    def getSize(self):
        return resLong, resShort
    
    def close(self):
        self.slm_lib.Delete_SDK()
        
class PCIe_beamshaper_core(PCIe_SLM):
    """ 
    Contains the super core code that is required. In most cases, this code is ran on the server!
    For reasons that are specific to how I want to reduce the latency, the rotation is done on the "core" side.
    
    Deliberately wrote it as a "pure" class, so that it essentially didn't have any intrinsic properties.
    """
    def __init__(self):   
        super().__init__()
        self.slm = self # this is simply to preserve the syntax "self.slm.updateArray()"
    
    def get_img_apply(self, mags, phases, periodWidth, angle):
        img_rotated = get_rotated_img(mags, phases, periodWidth, angle)
        self.slm.updateArray(img_rotated)

class PCIe_beamshaper():
    """
    The code of this is EXACTLY the same as the old "Beamshaper" class from "beamshaper.py",
    except 
    
    1) some of the attributes from the __init__ class are moved to the parent class
    "PCIe_SLM". Once this class is working, I want to change the code such that we can reuse 
    this class for both SLMs.
    
    2) the get_img function of this PCIe_beamshaper does not transpose its output. This is
    because this beamshaper is not mounted in a rotated position. Ideally, the transpose
    would move to the updateArray function in the Parent class, but somehow the transpose
    operation does not work there
    
    3) in "load_magnitude_calibration", the "PCIe_beamshaper_calibration.npz" instead of the
    "beamshaper_calibration.npz" is loaded, and the calibration works minimally different.
    
    4) the sawtooth/blazed grating points in the "other direction"
    
    5) 8/29/2023: The get_img function was completely updated to run faster
    
    6) 8/30/2023: delta_1_SLM, the misalignment variable, which was not used anymore, was
    completely purged from the code
    
    7) 9/1/2023: replace phase_quad and phase_cubic by delta_xx_xx in the calibration_dict to 
    reproduce k-x coupling observed in experiment
    
    """ 
    def __init__(self, 
                 calibration_dict = dict(
                     beamshaper_angle = 0,
                     phase_linear = 0/u.mm,
                     delta_3B_2F = 0*u.mm, 
                     delta_2B_1F = 0*u.mm, 
                     delta_1B_SLM = 0*u.mm, 
                     delta_3F_chip = 0*u.mm,
                 ), 
                 period_width = 16, 
                 mode = "multiuser",
                 load_calibration_flag=True,
                ):   
        """
        There are 3 modes that are supported on the beamshaper!
        multiuser: This assumes that a server kernel must be running!
        local: This means that no other instance of beamshaper is allowed at a given time (old code)
        offline: No beamshaper is connected -> Still useful for getting functions from beamshaper
        """
        self.periodWidth = period_width #hardcode for now
        self.load_calibration(calibration_dict)
        self.mode = mode
    
        self.lambda0 = 1.55*u.um
        self.k = 2*np.pi / self.lambda0
        
        self.f1 = 500*u.mm
        self.f2 = 200*u.mm
        self.f3 = 4.5*u.mm*(1.05) #artificially adding a magnification here! To compensate for the 5% deviation that we found via a measurement that we recently did
        
        self.aperture_L1 = 2*u.imperial.inch
        self.aperture_L2 = 22*u.mm
        self.aperture_L3 = 26.5*u.mm
        
        self.x_high_res = np.linspace(-(self.f1*self.lambda0/pixelPitch/2).to('mm').value, 
                                      (self.f1*self.lambda0/pixelPitch/2).to('mm').value, 
                                      60000)*u.mm
        
        if mode == "local":
            self.beamshaper_core = PCIe_beamshaper_core()
        if load_calibration_flag:
            self.load_magnitude_calibration()
        
    def load_calibration(self, calibration_dict):
        self.angle = calibration_dict['beamshaper_angle']
        self.phase_linear = calibration_dict['phase_linear'] #this is the shift in the frequency domain, to align with DMD
        
        self.delta_1B_SLM = calibration_dict['delta_1B_SLM']
        self.delta_2B_1F = calibration_dict['delta_2B_1F']
        self.delta_3B_2F = calibration_dict['delta_3B_2F']
        self.delta_3F_chip = calibration_dict['delta_3F_chip']
        
    def apply_mags_phases(self, mags, phases, linearize = True, debug=True):
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
        
        if any(mags < - 1e-10):
            print('One of the SLM magnitude is smaller than 0! Check normalization.')
            
        if linearize:
            mags = self.mag_des_2_mag_app(mags)
            
        ### Now run the code!###
        if self.mode == "offline":
            pass
        
        args = [mags, phases, self.periodWidth, self.angle]
        if self.mode == "local":
            self.beamshaper_core.get_img_apply(*args)
            
        if self.mode == "multiuser":
            client.run_command("beamshaper_core.get_img_apply", args)
            
        ### Now if want to debug the pattern that is applied:
        if debug:
            img_rotated = get_rotated_img(mags, phases, self.periodWidth, self.angle)
            self.img_applied = img_rotated #img_applied refers to the final image that is applied to the SLM.
            self.mags_applied = mags
            self.phases_applied = phases
        
    def apply(self, camps, linearize = True, debug=False):
        """
        camps: Complex amplitudes of the field.
        """
        mags = np.abs(camps)
        phases = -np.angle(camps) #######CCCCCCARRRRRRRRRAZZZZZZZZZY hack 2023-09-13, to change sign of fx. Please note that this is note rigorously derived, if possible, let's figure this out from first principles....self.
        self.apply_mags_phases(mags, phases, linearize = linearize, debug=debug)
    
    def phase_warning(self, mags, phases):
        # an indicator function that can be used to judge whether a particular output
        # requires a too high frequency phase pattern on the SLM. 
        # the closer to zero, the better, the closer to 1, the worse
        mags_norm = mags / mags.max()
        phases_unwrapped = np.unwrap(phases).value
        indicator = np.diff(phases_unwrapped) * mags_norm[1:] / np.pi
        return np.abs(indicator).max()

    def camps_chip_to_camps_SLM(self, x_chip, camps_chip, normalize = True):
        camps_chip = u.Quantity(camps_chip)

        calibration_vector = np.exp(1j*2*np.pi*(self.phase_linear*self.x_bs))
        camps_chip = camps_chip*calibration_vector
        
        x_3F, camps_3F = diffraction.free_space_propagation_backwards(
            x_chip, camps_chip, d = self.delta_3F_chip, k = 2*np.pi/self.lambda0)
        x_3B, camps_3B = diffraction.focal_plane_to_focal_plane_backwards(
            x_3F, camps_3F, f = self.f3, lambda0 = self.lambda0)
        x_2F, camps_2F = diffraction.free_space_propagation_backwards(
            x_3B, camps_3B, d = self.delta_3B_2F, k = 2*np.pi/self.lambda0)

        x_2B, camps_2B = diffraction.focal_plane_to_focal_plane_backwards(
            x_2F, camps_2F, f = self.f2, lambda0 = self.lambda0)
        x_1F, camps_1F = diffraction.free_space_propagation_backwards(
            x_2B, camps_2B, d = self.delta_2B_1F, k = 2*np.pi/self.lambda0)

        x_1B, camps_1B = diffraction.focal_plane_to_focal_plane_backwards(
            x_1F, camps_1F, f = self.f1, lambda0 = self.lambda0)

        x_SLM, camps_SLM = diffraction.free_space_propagation_backwards(
            x_1B, camps_1B, d = self.delta_1B_SLM, k = 2*np.pi/self.lambda0)

        if normalize:
            camps_SLM /= (np.abs(camps_SLM).max() )

        return x_SLM, camps_SLM

    @property
    def x_bs(self):
        """
        Returns the x_bs vector that is usable for the construction of input fields!
        x_bs = x_3F in old notation.
        Must use this if want to have success with using the all-important camps_3F_to_camps_SLM function
        """
        x_SLM = shortAxis
        xmax = (self.lambda0 * self.f1 / np.diff(x_SLM)[0]).to(u.mm)
        xnew = (ft_f_axis(len(x_SLM), np.diff(x_SLM)[0])*self.lambda0*self.f1).to(u.mm)
        x_bs_val = xnew * self.f3 / self.f2
        return x_bs_val


    #Ok, this is something that I have to very careful about now...
    #I need to add the debug flag in here...
    def apply_Ein(self, Ein_x_bs, sleep_time, normalize=True, debug=False):
        _, Ein_x_SLM = self.camps_chip_to_camps_SLM(self.x_bs, Ein_x_bs, normalize=normalize)
        self.apply(Ein_x_SLM, debug=debug)
        # time.sleep(sleep_time)
        
    def load_magnitude_calibration(self, plot_flag = False):
        """
        2023-10-28: Rewrite of this code
        """
        data = np.load('tdwg/lib/PCIe_beamshaper_calibration.npz')
        
        power_meas_list = data["power_meas_list"]
        power_applied_list = data["power_applied_list"]
        
        pow_des_2_pow_app = UnivariateSpline(power_meas_list, power_applied_list, s = 1e-3)
        
        def mag_des_2_mag_app(mag_des):
            pow_des = mag_des**2
            pow_app = pow_des_2_pow_app(pow_des)
            pow_app = np.clip(pow_app, 0, 1)
            mag_des = np.sqrt(pow_app)
            return mag_des
            
        self.mag_des_2_mag_app = mag_des_2_mag_app
        
        if plot_flag:
            power_desired_fine = np.linspace(0, 1.0, 1000)
            
            plt.figure(figsize=(5, 3))
            plt.plot(power_applied_list, power_meas_list, ".")
            plt.plot(power_applied_list, power_applied_list)
            plt.xlabel("Applied Power (a.u.)")
            plt.ylabel("Measured Power (a.u.)")
            plt.plot(pow_des_2_pow_app(power_desired_fine), power_desired_fine)
            plt.axhline(0)
            plt.axhline(1)

            
### The following code is regarding calibration of the beamshaper - Also verifying how well the calibration is going
    def calibrate_beamshaper(self, client, dmd, linecam, calibration_dict, input_flag=True, plot_flag=True):
        """
        This code performs the calibration with different "slits" that translates on the  the SLM plane. Using the average of these, it collects the calibration data for the grating efficiency.
        """
        power_applied_list = np.linspace(0.0, 1, 15)

        with client.locked():
            img_on = generate_all_on()
            dmd.apply_image_hold(img_on)
            time.sleep(0.5)

            power_mat = []
            for i in range(10):
                power_list = [0.0] #hard set first element to be zero
                for power_applied in power_applied_list[1:]:
                    normalize = True
                    sleep_time = 0.05

                    camps_SLM = np.ones(len(self.x_bs))
                    camps_SLM[i*100:(i+1)*100] = 1.0
                    camps_SLM = np.sqrt(power_applied)*camps_SLM
                    camps_SLM = u.Quantity(camps_SLM)

                    self.apply(camps_SLM, linearize=False)
                    time.sleep(sleep_time)


                    Iout_x_exp = linecam.get_output() - calibration_dict['cam_background']
                    power = np.sum(Iout_x_exp)
                    power_list.append(power)

                power_meas_list = np.array(power_list)/np.max(power_list)
                power_mat.append(power_meas_list)
                
        power_meas_list = np.mean(power_mat, axis=0)
        power_meas_list = power_meas_list/np.max(power_meas_list)
        power_meas_list[-1] = 1.0
        
        if plot_flag:
            plt.figure(figsize=(5,2.5))
            for power_meas_list in power_mat: 
                plt.plot(power_applied_list, power_meas_list, color="k", alpha=0.15)
            plt.xlabel("Applied Power (a.u.)")
            plt.ylabel("Measured Power (a.u.)")
            plt.show()
                
        if input_flag:
            # Get user input
            response = input("Would you like to overwrite the beam shaper calibration? (yes/no) ")

            # Check if response is "yes"
            if response.lower() == "yes":
                np.savez("tdwg/lib/PCIe_beamshaper_calibration.npz", 
                         power_meas_list=power_meas_list, power_applied_list=power_applied_list)
                print("Calibration overwritten.")
                
                self.load_magnitude_calibration(plot_flag=True)
            else:
                print("Calibration not overwritten.")
                
    def check_linearity(self, Ein_bs, client, dmd, linecam, calibration_dict):
        def get_power(power_applied):
            _, Ein_x_SLM = self.camps_chip_to_camps_SLM(self.x_bs, Ein_bs, normalize=True)
            Ein_x_SLM = np.sqrt(power_applied)*Ein_x_SLM

            self.apply(Ein_x_SLM, linearize=True)
            time.sleep(0.05)

            Iout_x_exp = linecam.get_output() - calibration_dict['cam_background']
            power = np.sum(Iout_x_exp)
            return power, Iout_x_exp
        
        power_app_list = np.linspace(0.001, 1, 20)

        with client.locked():
            img_on = generate_all_on()
            dmd.apply_image_hold(img_on)
            time.sleep(0.5)

            power_list, Iout_x_exp_list = sweep_zip(get_power, power_app_list)
            
        power_meas_list = power_list/np.max(power_list)
        
        plt.figure(figsize=(5, 3))
        plt.plot(power_app_list, power_meas_list)
        plt.plot(power_app_list, power_app_list, "--", color="gray")
        plt.xlabel("Applied Power (a.u.)")
        plt.ylabel("Measured Power (a.u.)")
        plt.show()
        
        return power_app_list, power_meas_list, Iout_x_exp_list
        
        