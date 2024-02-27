###### TODO #########
# Can add the beamshaper and stuff too
# Too lazy, will do later... That also involves a different set of code... Probably need a new class, now that I think about it more. 
# Wiped out the code recently and reinstated Code: McMahon lab Dropbox\peter mcmahon\to232\2D-waveguide\2023-05-04 physics based digital twin of LN chip version! Now will add other stuff

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()

import torch
from scipy.interpolate import interp2d
import torch.nn.functional as F
from tdwg.lib.simulation import WaveguideSimulation

from tdwg.lib.io_utils import *
crazy_reload("tdwg.lib.simulation", globals())

class Exp_Sim_Converter():
    """
    Some naming conventions that are useful to be aware of when reading the source code.
    # mod, to represent modulations of the index that is between 0 and 1!
    # mod_exp would be the DMD_image_up to 
    # mod_exp_window would be the DMD image with the start_pad_ind and end_pad_ind removed
    # mod_sim_whole would be the "whole" thing, including the front_dist and back_dist
    # mod_sim would be the modulation around the DMD region only
    # mod_sim_window would be the modulation in the simulation axis that is "windowed" around the DMD window

    #there are going to be slips involved, becase the code is not perfect. That said, it is on the order of 2um, which I can't calibrate for that accurately anyways! So just ignore this "bug"
    #for instance self.wg.x_axis and self.x_axis_exp_crop are not "aligned", but I think that should be fine...
    """
    def __init__(self, calibration_dict, Ncom=20):
        self.Ncom = Ncom
        self.load_calibration(calibration_dict)

    def load_calibration(self, calibration_dict):
        """
        This is to load the calibration dictionary, which is a dictionary that contains all the calibration parameters.
        """
        self.Lx = calibration_dict["Lx"]
        self.Nx = calibration_dict["Nx"]
        self.Nz = calibration_dict["Nz"]
        Ncom = self.Ncom
        
        self.dn_norm_func = lambda x : calibration_dict["dn_rescale_factor"]*dn_norm_func(calibration_dict["LED_power_factor"]*x)

        self.diffusion_length = calibration_dict["diffusion_length"]
        self.ind_beam_center = calibration_dict["ind_beam_center"]

        self.DMD_res_x = calibration_dict["DMD_res_x"]
        self.DMD_res_z = calibration_dict["DMD_res_z"]
        self.neff = calibration_dict["neff"]

        self.img_exp_pitch = calibration_dict["img_exp_pitch"]
        self.front_dist = calibration_dict["front_dist"]
        self.back_dist = calibration_dict["back_dist"]
        self.front_ind_pad = calibration_dict["front_ind_pad"]
        self.back_ind_pad = calibration_dict["back_ind_pad"]

        self.Lz = self.front_dist + self.back_dist + self.DMD_res_z*self.img_exp_pitch
        self.wg = WaveguideSimulation(self.neff, self.Lx, self.Lz, self.Nx, self.Nz, diffusion_length = self.diffusion_length, Ncom=Ncom)
        
        # can move to separate function, but I think it is fine in this case?
        z_window_start = self.front_dist + self.front_ind_pad*self.img_exp_pitch
        z_window_end = self.wg.Lz - self.back_dist - self.back_ind_pad*self.img_exp_pitch

        zind_sim_window_start = self.wg.z2ind(z_window_start)
        zind_sim_window_end = self.wg.z2ind(z_window_end)
        #these are useful objects
        mod_sim_window_shape = [self.wg.Nx, zind_sim_window_end - zind_sim_window_start]
        z_axis_sim_window = self.wg.z_axis[zind_sim_window_start:zind_sim_window_end]

        #save some of these variables
        self.mod_sim_window_shape = mod_sim_window_shape
        self.z_axis_sim_window = z_axis_sim_window
        self.z_window_start = z_window_start
        self.z_window_end = z_window_end
        self.zind_sim_window_start = zind_sim_window_start
        self.zind_sim_window_end = zind_sim_window_end

        ####### Below are setting up the coordinate systems of the experiment ########
        #First get the quantities related to the z_axis
        front_dist_ind_len = int((self.front_dist/self.img_exp_pitch).decompose()) #numer of indices for the length
        back_dist_ind_len = int((self.back_dist/self.img_exp_pitch).decompose()) # not using ind intensionally, as that is used to refer to discrete indices instead of actually length usually. 
        z_axis_exp_whole = np.arange(0, self.DMD_res_z+front_dist_ind_len+back_dist_ind_len)*self.img_exp_pitch #this corresponds to the z_axis of the "whole"

        #This is to get the items related to the x_axis
        x_crop_lim = self.wg.x_axis[-1]
        x_axis_exp_img = (np.flip(1 + np.arange(0, self.DMD_res_x)) - (self.DMD_res_x-self.ind_beam_center))*self.img_exp_pitch #this is the xaxis of the entire DMD - so it is way larger than the 
        x_exp_img_2ind = lambda x: np.argmin(abs(x_axis_exp_img - x))
        ind_x_crop_lim_pos = x_exp_img_2ind(x_crop_lim)
        ind_x_crop_lim_neg = x_exp_img_2ind(-x_crop_lim)
        #because the x-coordinate is flipped, that is why the indexing is going the other way! Crazy...
        x_axis_exp_crop = x_axis_exp_img[ind_x_crop_lim_pos:ind_x_crop_lim_neg]

        self.front_dist_ind_len = front_dist_ind_len
        self.back_dist_ind_len = back_dist_ind_len
        self.z_axis_exp_whole = z_axis_exp_whole
        self.x_axis_exp_crop = x_axis_exp_crop

        self.ind_x_crop_lim_pos = ind_x_crop_lim_pos
        self.ind_x_crop_lim_neg = ind_x_crop_lim_neg

        z_exp_whole_2_ind = lambda z: np.argmin((self.z_axis_exp_whole-z)**2)
        z_axis_exp_window = self.z_axis_exp_whole[z_exp_whole_2_ind(self.z_window_start) : z_exp_whole_2_ind(self.z_window_end)]
        self.z_axis_exp_window = z_axis_exp_window

    def img_2_mod_sim_whole(self, img):
        """ 
        img: The image that is applied to the DMD
        returns mod_sim_whole: a modulation pytorch tensor that specifies how much modulation of the index (normalized between 0 and 1) is applied in the coordinates of the simulation!
        """
        mod_exp = img/255

        if self.front_dist_ind_len < 0 and self.back_dist_ind_len<0:
            mod_exp_whole = mod_exp[:, -self.front_dist_ind_len:self.back_dist_ind_len]
            
        elif self.front_dist_ind_len > 0 and self.back_dist_ind_len > 0:
            mod_exp_whole = np.pad(mod_exp, ((0, 0), (self.front_dist_ind_len, self.back_dist_ind_len)), mode="constant", constant_values=0)
            
        else:
            raise Exception("I haven't yet coded up the case of the DMD partially covering the chip! Ask Hiro to write more code...")
        
        mod_exp_crop = mod_exp_whole[self.ind_x_crop_lim_pos:self.ind_x_crop_lim_neg, :]

        # Implicit in the naming for mod_sim_whole, is that x_axis_crop is used to conform to the sim coordinate boundaries.
        interp_func = interp2d(self.z_axis_exp_whole.to("um").value, self.x_axis_exp_crop.to("um").value, mod_exp_crop, kind="linear", bounds_error=False, fill_value=0)
        mod_sim_whole = interp_func(self.wg.z_axis.to("um").value, self.wg.x_axis.to("um").value)
        mod_sim_whole = torch.tensor(mod_sim_whole).float() #convert to tensor object at this point!
        return mod_sim_whole

    def mod_sim_whole_2_delta_n(self, mod_sim_whole):
        #Write the code with the invert!        
        mod_sim_whole_filtered = self.wg.smoothen_spatial_map(mod_sim_whole.T)
        delta_n = self.dn_norm_func(mod_sim_whole_filtered)
        return delta_n

    def mod_sim_window_2_mod_sim_whole(self, mod_sim_window):
        #code generated by chatGPT
        padding_top = self.zind_sim_window_start
        padding_bottom = self.wg.Nz - self.zind_sim_window_end

        mod_sim_whole = F.pad(mod_sim_window, (padding_top, padding_bottom, 0, 0), "constant", 0)
        return mod_sim_whole

    def mod_sim_window_2_img(self, mod_sim_window): 
        """ 
        Input: mod_sim_window, a pytorch tensor that represents the image to apply to DMD
        Ouput: The actual image that can be applied to the DMD
        Definitely the hardest function to write!
        """
        mod_sim_window_np = mod_sim_window.numpy()
        interp_func = interp2d(self.z_axis_sim_window.to("um").value, self.wg.x_axis.to("um").value, mod_sim_window_np, kind="linear", bounds_error=False, fill_value=0)

        # So there is some kind of bug with interp2d where it cannot deal with indices that go "backward", so for now, just use this hack to flip things! Look into this when I have internet connection
        mod_exp_window = interp_func(self.z_axis_exp_window.to("um").value, self.x_axis_exp_crop.to("um").value)
        mod_exp_window = np.flip(mod_exp_window, 0) #can't be bothered, let's just use this hack for now

        # Now pad things so they are of the right dimension
        mod_exp = np.pad(mod_exp_window, ((self.ind_x_crop_lim_pos, self.DMD_res_x-self.ind_x_crop_lim_neg), (self.front_ind_pad, self.back_ind_pad)), mode="constant", constant_values=0)

        img = (mod_exp*255).astype(np.uint8)
        return img

    def mod_sim_whole_2_mod_sim_window(self, mod_sim_whole):
        return mod_sim_whole[:, self.zind_sim_window_start:self.zind_sim_window_end]

    

from tdwg.lib.SRN_covered_LN_waveguide import WG_linear, imp_parallel, imp_series, imp_cap
from tdwg.lib.conductivity_fits import eps_r_a4

sigma0 = 7.41508871e-08
alpha = 9.26092992e-09
const = 7.68956977e-01
Vapplied = 1100
wg_electrical = WG_linear(d_co = 700e-9, d_pc = 4e-6, eps_pc=eps_r_a4)
ZC_pc = imp_cap(10, wg_electrical.C_pc)
Z_wg_electrical = torch.tensor(imp_series([imp_cap(10, wg_electrical.C_cl), imp_cap(10, wg_electrical.C_co)]))
Z_LN = torch.tensor(imp_cap(10, wg_electrical.C_co))

def r_eff_to_delta_n(r_eff, n0, E):
    return n0**3 / 2 * r_eff * E

def dn_func(intensity_LED):
    R_pc = 1 / (sigma0 + alpha * intensity_LED)
    Z_pc = 1/( 1/ZC_pc + 1/R_pc)
    return const * r_eff_to_delta_n(wg_electrical.r33, wg_electrical.n_co, Vapplied/wg_electrical.d_co*torch.abs(Z_LN / (Z_wg_electrical + Z_pc)))

#make a new function that takes in between 0 and 1
max_power = 67
dn_norm_func = lambda x: dn_func(x*max_power) - dn_func(0)