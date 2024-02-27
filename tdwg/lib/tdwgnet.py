"""
Classes:

TDwgExperiment
    Collects functions to run the experiment in different modes.

FineTunedSimulation
    Collects functions to run the simulation in different modes,
    plus parameters to fine-tune the simulation to agree better with experiments.

TDwgModel:
    Acts like a model in a neural network with trainable parameters (mod_sim_window)
    and a forward function (using the simulation [mode='insilico'] or experiment [mode='pat']).
    
TDwgLayer:
    Acts like a layer in a neural network with trainable parameters (mod_sim_window)
    and a forward function (using the simulation [mode='insilico'] or experiment [mode='pat']).
    By layer, it now includes the input and output modes! So it takes in quantities of an appropriate dimension!
"""

import numpy as np
import time
import copy
import torch
from torch import nn
import astropy.units as u

from tdwg.lib.simulation import interp1d
import tdwg.lib.ftutils_torch as ftutils_torch
from tdwg.lib.ftutils_torch import fft_centered_ortho, ifft_centered_ortho, ft_t_axis, ft_f_axis
import tdwg.lib.pnn_utils as pnn_utils 
from tdwg.lib.pat import make_pat_func
from tdwg.lib.mode_utils import make_gaussian_modes, make_boxed_modes
import multiprocess as mp
from tdwg.lib.noise import make_pink_noise

######## Following code is for making the code run super fast#########
def init_worker():
    from tdwg.lib.exp_sim_converter import Exp_Sim_Converter
    from scipy.interpolate import interp1d
    global interp1d
    global beamshaper_offline
    global Ein_sim_amp_cali
    global wg_x_axis
    
    import pickle
    with open('calibration_dict.pkl', 'rb') as file:
        calibration_dict = pickle.load(file)
    
    converter = Exp_Sim_Converter(calibration_dict, Ncom=3)
    wg = converter.wg
    wg_x_axis = wg.x_axis
    Ein_sim_amp_cali = calibration_dict["Ein_sim_amp_cali"]

    from tdwg.lib.PCIe_beamshaper import PCIe_beamshaper
    beamshaper_offline = PCIe_beamshaper(mode="offline", calibration_dict=calibration_dict)

def worker_task(Ein_sim):
    Ein_sim = Ein_sim.detach().cpu().numpy() #conversion to numpy
    Ein_sim = Ein_sim*Ein_sim_amp_cali #calibrate the object!
    input_beam_bs = interp1d(wg_x_axis, Ein_sim, bounds_error=False, fill_value=0)(beamshaper_offline.x_bs)
    
    beamshaper_offline.apply_Ein(input_beam_bs, 0.0, debug=True)
    return beamshaper_offline.img_applied

class TDwgExperiment():
    """
    Collects functions to run the experiment in different modes:
    
    run_exp:
        take a 1024x768 dimensional DMD image and a wg.Nx dimensional 
        Ein_sim and directly returns camera output
    
    run_exp_fixed_DMD
        take a a wg.Nx dimensional Ein_sim and directly returns camera output.
        Faster than run_exp but requires that the DMD is manually set beforehand.
        
    run_exp_units_sim:
        same as run_exp but converts the camera output to the dimensions used in 
        wg simulation object.
        
    run_exp_fixed_DMD_units_sim:
        same as run_exp_fixed_DMD but converts the camera output to the dimensions 
        used in wg simulation object.
        
    convert_exp_units_to_sim:
        converts camera outputs (usually 630-dim) to the dimensions used in 
        wg simulation object (usually 1024-dim).
        
    """
    def __init__(self, 
                 converter, linecam, beamshaper, dmd, client,
                 calibration_dict = dict(cam_background=0, Ein_sim_amp_cali=1)
                ):
        self.converter = converter
        self.wg = converter.wg
        self.linecam = linecam
        self.beamshaper = beamshaper 
        self.dmd = dmd
        self.client = client
        self.x_bs = beamshaper.x_bs
        
        self.load_calibration(calibration_dict)
        self.create_pool()
        
    def load_calibration(self, calibration_dict):
        self.background = calibration_dict['cam_background']
        self.Ein_sim_amp_cali = calibration_dict['Ein_sim_amp_cali']
        self.bs_sleep = calibration_dict["bs_sleep"]
        self.DMD_sleep = calibration_dict["DMD_sleep"]
        
    def update_cam_background(self, lock_flag=True):
        if lock_flag:
            self.client.lock()
        self.beamshaper.apply(u.Quantity(np.zeros(1024, dtype = complex)))
        time.sleep(0.03)
        self.background = self.linecam.get_output()
        if lock_flag:
            self.client.unlock()
        
    def run_exp_fixed_DMD_expcoord(self, Ein_bs: np.ndarray) -> np.ndarray:
        self.beamshaper.apply_Ein(Ein_bs, sleep_time=0.0)
        time.sleep(self.bs_sleep)

        Iout_x_exp = self.linecam.get_output() - self.background
        return Iout_x_exp

    def run_exp_fixed_DMD(self, Ein_sim: torch.Tensor) -> torch.Tensor:
        Iin_sim_norm = (Ein_sim.abs()**2).sum()
        Ein_sim = Ein_sim.detach().cpu().numpy() #conversion to numpy
        Ein_sim = Ein_sim*self.Ein_sim_amp_cali #calibrate the object!
        input_beam_bs = interp1d(self.wg.x_axis, Ein_sim, bounds_error=False, fill_value=0)(self.x_bs)

        Iout_x_exp = self.run_exp_fixed_DMD_expcoord(input_beam_bs) #this is in np form still
        
        Iout_x_sim = np.interp(self.wg.x_axis, self.linecam.x_axis, Iout_x_exp, left = 0, right = 0)
        Iout_x_sim = Iout_x_sim*Iin_sim_norm.numpy()/np.sum(Iout_x_sim)
        Iout_x_sim = torch.from_numpy(Iout_x_sim) #now convert to torch
        return Iout_x_sim

    def run_exp(self, img, Ein_sim: torch.Tensor) -> torch.Tensor:
        with self.client.locked():
            self.dmd.apply_image_hold(img, bitDepth=8)
            time.sleep(self.DMD_sleep) #This also should be a parameter ideally.

            return self.run_exp_fixed_DMD(Ein_sim)

    def run_exp_Ein_list(self, img, Ein_sim_list: torch.Tensor, fast_flag=True) -> torch.Tensor:
        with self.client.locked():
            self.dmd.apply_image_hold(img, bitDepth=8)
            time.sleep(self.DMD_sleep) #This also should be a parameter ideally.
    
            if fast_flag:
                Iout_x_sim_list = self.run_exp_fixed_DMD_list_fast(Ein_sim_list) 
            else:
                Iout_x_sim_list = [self.run_exp_fixed_DMD(Ein_sim) for Ein_sim in Ein_sim_list]

        Iout_x_sim_list = torch.vstack(Iout_x_sim_list)
        return Iout_x_sim_list

    def forward(self, Ein_x_sim: torch.Tensor, mod_sim_window):
        """
        Purpose of function is to simplify the API. 
        Also to make the clipping and conversion happen here
        """
        mod_sim_window_clip = mod_sim_window.clip(0, 1).detach().cpu()
        img = self.converter.mod_sim_window_2_img(mod_sim_window_clip)

        Iout_x_exp = self.run_exp_Ein_list(img, Ein_x_sim)
        return Iout_x_exp
         

    ############ All the rest is for "fast" code ###########
    def create_pool(self):
        if self.linecam.mode != "offline":
            self.pool = mp.Pool(8, initializer=init_worker)
        
    def run_exp_fixed_DMD_list_fast(self, Ein_sim_list):
        bs_img_list = self.pool.map(worker_task, Ein_sim_list) #First in a batch, get all of the images!
        Iout_x_exp_list = []

        self.beamshaper.beamshaper_core.slm.updateArray(bs_img_list[0])
        time.sleep(self.bs_sleep)
        
        for bs_img in bs_img_list[1:]:
            self.linecam.line_camera_core.fg.wait_for_frame('now')
            self.beamshaper.beamshaper_core.slm.updateArray(bs_img)
            Iout_x_exp_list.append(self.linecam.get_output(trigger_local=False))
            # time.sleep(4e-3)

        self.linecam.line_camera_core.fg.wait_for_frame('now')
        Iout_x_exp_list.append(self.linecam.get_output(trigger_local=False))
        
        Iout_x_sim_list = []
        for (Ein_sim, Iout_x_exp) in zip(Ein_sim_list, Iout_x_exp_list):
            Iin_sim_norm = (Ein_sim.abs()**2).sum()
            Iout_x_exp = Iout_x_exp - self.background
            Iout_x_sim = np.interp(self.wg.x_axis, self.linecam.x_axis, Iout_x_exp, left = 0, right = 0)
            Iout_x_sim = Iout_x_sim*Iin_sim_norm.numpy()/np.sum(Iout_x_sim)
            Iout_x_sim = torch.from_numpy(Iout_x_sim) #now convert to torch
            Iout_x_sim_list.append(Iout_x_sim)
        return Iout_x_sim_list#, dt_list

class FineTunedSimulation(nn.Module):
    """
    Collects functions to run the simulation in different modes,
    plus parameters to fine-tune the simulation to agree better with experiments:
    """
    def __init__(self, converter, train_flag = False):
        super().__init__()
        self.converter = converter
        self.wg = self.converter.wg
        
        # single number parameters
        self.fx_mod_per_mm = nn.Parameter(torch.tensor(0.), requires_grad = train_flag)
        self.displacement_in = nn.Parameter(torch.tensor(0.), requires_grad = train_flag)
        self.displacement_out = nn.Parameter(torch.tensor(0.), requires_grad = train_flag)

        # 1D parameters
        self.input_coupling = nn.Parameter(torch.ones(self.wg.Nx), requires_grad = train_flag)
        self.output_coupling = nn.Parameter(torch.ones(self.wg.Nx), requires_grad = train_flag)
        self.cam_background = nn.Parameter(torch.zeros(self.wg.Nx), requires_grad = False)

        # 2D parameters
        self.background_mod = nn.Parameter(torch.zeros(self.wg.Nz, self.wg.Nx), requires_grad = train_flag)
        self.modulation_efficiency = nn.Parameter(torch.ones(self.wg.Nz, self.wg.Nx), requires_grad = train_flag)
#         self.modulation_efficiency = nn.Parameter(torch.tensor(1.), requires_grad = train_flag)
        
        self.background_delta_n = self.converter.dn_norm_func(self.background_mod)
        self.register_buffer('x_axis_mm', torch.from_numpy(self.wg.x_axis.to('mm').value))
        
    def run_sim_fixed_DMD(self, Ein_sim: torch.Tensor, fast_flag=True):
        """
        This codes runs different input vectors, assuming the DMD is fixed at some point. 
        """
        Ein_sim = Ein_sim*torch.exp(1j*2*np.pi*self.fx_mod_per_mm*self.x_axis_mm)
        
        self.background_delta_n = self.converter.dn_norm_func(self.background_mod)

        Ein_sim = displace(self.x_axis_mm, Ein_sim, self.displacement_in)
        Ein_sim = Ein_sim * self.input_coupling

        delta_n = self.modulation_efficiency * self.wg.delta_n.to(Ein_sim.device) + self.background_delta_n
        if fast_flag:
            Eout_x_sim = self.wg.run_simulation(Ein_sim, delta_n)
        else:
            Eout_x_sim = self.wg.run_simulation_slow(Ein_sim, delta_n)
        
        Eout_x_sim = Eout_x_sim * self.output_coupling
        Eout_x_sim = displace(self.x_axis_mm, Eout_x_sim, self.displacement_out)
        Iout_x_sim = Eout_x_sim.abs()**2 + self.cam_background
        
        return Iout_x_sim 

    def run_sim_slow(self, img, Ein_sim: torch.tensor):
        mod_sim_whole = self.converter.img_2_mod_sim_whole(img)
        delta_n = self.converter.mod_sim_whole_2_delta_n(mod_sim_whole)
        self.wg.set_delta_n(delta_n)
        Eout_x_sim = self.run_sim_fixed_DMD(Ein_sim, fast_flag=False)
        return Eout_x_sim, copy.copy(self.wg)

    def run_sim(self, img, Ein_sim: torch.Tensor):
        mod_sim_whole = self.converter.img_2_mod_sim_whole(img)
        delta_n = self.converter.mod_sim_whole_2_delta_n(mod_sim_whole)
        self.wg.set_delta_n(delta_n)

        return self.run_sim_fixed_DMD(Ein_sim)
    
    def run_sim_from_mod_sim_window(self, Ein_sim: torch.Tensor, mod_sim_window, fast_flag=True):
        mod_sim_window_clip = mod_sim_window.clip(0, 1)
        mod_sim_whole = self.converter.mod_sim_window_2_mod_sim_whole(mod_sim_window_clip)
        delta_n = self.converter.mod_sim_whole_2_delta_n(mod_sim_whole)
        self.wg.set_delta_n(delta_n)
        return self.run_sim_fixed_DMD(Ein_sim, fast_flag=fast_flag)

    def forward(self, Ein_sim: torch.Tensor, mod_sim_window):
        return self.run_sim_from_mod_sim_window(Ein_sim, mod_sim_window, fast_flag=True)
        
def displace(x_axis, field_x, d):
    field_f = fft_centered_ortho(field_x)
    f_axis = ft_f_axis(len(x_axis), x_axis[1]-x_axis[0], device = x_axis.device)
    field_f *= torch.exp(-1j*2*torch.pi*d*f_axis)
    field_x_displaced = ifft_centered_ortho(field_f)
    return field_x_displaced
    
class TDwg(nn.Module):
    """
    Acts like a pytorch model in a neural network with trainable parameters (mod_sim_window)
    and a forward function (using the simulation [mode='insilico'] or experiment [mode='pat']).
    The most important variable is mod_sim_window, which is a tensor from which the 
    DMD image can be constructed.
    forward_mode and forward_physical nominally return the exact same output, bar normalization,
    and any simulation-experiment mismatches (wavefront damage, calibration, ...)
    """
    def __init__(self, tdwg_exp, tdwg_sim, mode, random_flag=False):
        super().__init__()
        
        if mode not in ["pat", "insilico"]:
            raise ValueError("mode has to be 'pat' or 'insilico'. '{mode}' was given.")
            
        self.tdwg_sim = tdwg_sim
        self.tdwg_exp = tdwg_exp
        self.mode = mode
        self.f_pat = make_pat_func(self.tdwg_exp.forward, self.tdwg_sim.forward)
        
        mod_sim_window_tensor = 0.5*torch.ones(self.tdwg_sim.converter.mod_sim_window_shape, dtype=torch.float32)

        if random_flag==True:
            mod_sim_window = make_pink_noise(self.tdwg_sim.converter.z_axis_sim_window, self.tdwg_sim.wg.x_axis, 
                                     0.1, 30*u.um, 100*u.um, skew=50)[0]
            mod_sim_window = mod_sim_window.real
            mod_sim_window = mod_sim_window - mod_sim_window.min() + 0.2*np.ones_like(mod_sim_window)
            mod_sim_window_tensor = torch.from_numpy(mod_sim_window).T

        self.mod_sim_window = pnn_utils.Parameter(mod_sim_window_tensor, limits=[0.1, 0.9], requires_grad= True)
        

    def forward(self, x):
        if self.mode=="pat":
            return self.f_pat(x, self.mod_sim_window)
        elif self.mode=="insilico":
            return self.tdwg_sim.forward(x, self.mod_sim_window)
        else:
            raise ValueError("mode has to be 'pat' or 'insilico'. '{mode}' was given.")

class TDwgLayer(nn.Module):
    def __init__(self, tdwg_exp, tdwg_sim, mode, Nin, Nout, w0_in, xmode_in_lim, xmode_out_lim, seed="identity", random_flag=False):
        super().__init__()
        
        self.tdwg = TDwg(tdwg_exp, tdwg_sim, mode, random_flag=random_flag) #this way a new one will be instantiated for each layer!
        x_axis = tdwg_sim.wg.x_axis
        input_modes = make_gaussian_modes(x_axis, Nin, xmode_in_lim, w0_in)
        output_modes = make_boxed_modes(x_axis, Nout, xmode_out_lim)

        if seed == 'identity':
            input_perm = torch.arange(Nin)
            output_perm = torch.arange(Nout)
        else:
            torch.manual_seed(seed)
            input_perm = torch.randperm(Nin)
            output_perm = torch.randperm(Nout)
        
        self.register_buffer("input_modes", input_modes)
        self.register_buffer("output_modes", output_modes)
        self.register_buffer("input_perm", input_perm)
        self.register_buffer("output_perm", output_perm)

    def forward(self, x):
        x = x.to(torch.complex128)
        x = x[:, self.input_perm]

        # propagation through experiment
        input_beams = x@self.input_modes
        output_intensity = self.tdwg.forward(input_beams)

        # "Fan-in" with output_modes
        y = output_intensity@self.output_modes.T
        y = y[:, self.output_perm]

        self.save_dict = dict(input_beams=input_beams.detach(), output_intensity=output_intensity.detach())
        return y