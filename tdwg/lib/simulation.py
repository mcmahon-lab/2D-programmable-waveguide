import torch
import torch.fft as fft
from fft_conv_pytorch import fft_conv, FFTConv2d
import numpy as np
from scipy.stats import multivariate_normal
import tdwg.lib.ftutils_np as ftutils
import matplotlib.pyplot as plt
from tdwg.lib.DMD_patterns import generate_fill_factor_mask
from tdwg.lib.ftutils_torch import fft_centered_ortho, ft_f_axis, ifft_centered_ortho
import astropy.units as u
import copy
from scipy.interpolate import interp1d, interp2d
from tdwg.lib.DMD_patterns import invert_pattern

class WaveguideSimulation():
    def __init__(self, neff, Lx, Lz, Nx, Nz, diffusion_length, Ncom=1):
        #all the units for length are in microns!
        self.n =  neff # effective refractive index for the slab waveguide
        self.Lx = Lx # length of transverse direction
        self.Lz = Lz # beam propagation distance
        
        self.Nx = Nx #Be power of 2 for FFT
        self.Nz = Nz
        
        self.lam0 = 1.55*u.um #wavelength of the fundamental
        
        self.Ncom = Ncom # wavefront is saved every Ncom integration steps

        self.k0 = 2*np.pi/self.lam0 # k-number in free space

        self.dx = self.Lx/(self.Nx-1) # stepsize in transverse dimension
        
        self.x_axis = ftutils.ft_t_axis(self.Nx, self.dx)
        self.z_axis = np.linspace(0, self.Lz, self.Nz) # array holding z-coordinates in um

        self.dz = self.z_axis[1]-self.z_axis[0] # stepsize in propagation direction

        self.fx_axis = ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis

        # The following defines *dimensionless* quantities, which are used in the simulation internal loop!
        difr_list = np.fft.fftshift(np.exp((-1j*self.kx_axis**2/(2*self.n*self.k0)*self.dz).decompose().value))
        self.difr_list = torch.tensor(difr_list)
        self.k0dz = (self.k0*self.dz).decompose().value

        self.set_diffusion_length(diffusion_length)

        self.x2ind = lambda x: np.argmin(np.abs(self.x_axis-x))
        self.z2ind = lambda z: np.argmin(np.abs(self.z_axis-z))
        self.zlist2ind = lambda z: np.argmin(np.abs(self.z_list-z))

        
        # create multivariate gaussian to emulate carrier diffusion
        self.kernel = torch.FloatTensor(get_gaussian_kernel(6*self.nz_kernel+1, 6*self.nx_kernel+1)).unsqueeze(0).unsqueeze(0)

    def set_diffusion_length(self, diffusion_length):
        self.diffusion_length = diffusion_length
        self.nx_kernel = int(np.round((self.diffusion_length/self.dx).decompose()))
        self.nz_kernel = int(np.round((self.diffusion_length/self.dz).decompose()))

    def set_delta_n(self, delta_n):
        if delta_n.shape != (self.Nz, self.Nx):
            raise ValueError('spatial_map has wrong shape, should be [self.Nz, self.Nx]')
        self.delta_n = delta_n

    def smoothen_spatial_map(self, spatial_map, padding_mode = 'reflect'):
        # convolves a spatial map (such as a delta_n) with a gaussian kernel of standard deviation equal to the carrier diffusion length (in um). The input should have size of [self.Nz, self.Nx]

        if spatial_map.shape != (self.Nz, self.Nx):
            raise ValueError('spatial_map has wrong shape, should be [self.Nz, self.Nx]')

        # pad delta n and convolve it with gaussian kernel
        spatial_map = fft_conv(spatial_map.unsqueeze(0).unsqueeze(0), self.kernel.to(spatial_map.device), bias=None, padding = [3*self.nz_kernel,3*self.nx_kernel], padding_mode=padding_mode)
        return spatial_map.squeeze(0).squeeze(0)

    def run_simulation(self, a, delta_n = None):
        """
        Use this if you want code to be fast!
        a: The input beam!
        """
        if delta_n is None: delta_n = self.delta_n.to(a.device)
        else: self.delta_n = delta_n.to(a.device)
            
        delta_n_term = torch.exp(1j*self.k0dz*delta_n) #takes about 40us to run, so don't worry about optimizing it!
        difr_list = self.difr_list.to(a.device)

        for delta_n_term_slice in delta_n_term:
            a = delta_n_term_slice * a
            ak = fft.fft(a)
            ak = difr_list * ak
            a = fft.ifft(ak)
        return a
    
    def run_simulation_slow(self, a, delta_n = None):
        """
        Around 2X slower than the fast version for Nz being 1000
        """
        self.a_list = []
        self.ak_list = []
        self.z_list = []

        if delta_n is None: delta_n = self.delta_n
        else: self.delta_n = delta_n
            
        delta_n_term = torch.exp(1j*self.k0dz*delta_n) 

        z = 0*u.um
        for (z_ind, delta_n_term_slice) in enumerate(delta_n_term):
            a = delta_n_term_slice * a
            ak = fft.fft(a)
            ak = self.difr_list.to(a.device) * ak
            a = fft.ifft(ak)

            if z_ind % self.Ncom == 0:
                self.a_list.append(a)
                self.ak_list.append(torch.fft.ifftshift(ak))
                self.z_list.append(copy.deepcopy(z))
            z += self.dz
        
        self.Emat_x = torch.stack(self.a_list)
        self.Emat_f = torch.stack(self.ak_list)
        self.z_list = u.Quantity(self.z_list)

        self.Eout_x = self.Emat_x[-1].detach().cpu().numpy()
        self.Eout_f = self.Emat_f[-1].detach().cpu().numpy()
        self.Iout_x = abs(self.Emat_x[-1].detach().cpu().numpy())**2
        self.Iout_f = abs(self.Emat_f[-1].detach().cpu().numpy())**2

        self.Ein_x = self.Emat_x[0].detach().cpu().numpy()
        self.Ein_f = self.Emat_f[0].detach().cpu().numpy()
        self.Iin_x = abs(self.Emat_x[0].detach().cpu().numpy())**2
        self.Iin_f = abs(self.Emat_f[0].detach().cpu().numpy())**2
        return a

    def new_wg_Nz(self, Nz_new, Ncom=1):
        z_axis_new = np.linspace(self.z_axis[0], self.z_axis[-1], Nz_new)
        wg_new = WaveguideSimulation(self.n, self.Lx, self.Lz, self.Nx, Nz_new, Ncom=Ncom)
            
        wg = self
        interp_func = interp2d(wg.x_axis.to("um").value, wg.z_axis.to("um").value, wg.delta_n.cpu().detach().numpy(), kind="linear", bounds_error=False, fill_value=0)
        delta_n_new = interp_func(wg.x_axis.to("um").value, z_axis_new.to("um").value)
        delta_n_new = torch.tensor(delta_n_new)
        wg_new.set_delta_n(delta_n_new)
        return wg_new

    def _plot_delta_n(self, xlim=200):
        wg = self
        plt.pcolormesh(wg.z_axis.to("mm").value, wg.x_axis.to("um").value, wg.delta_n.T.detach().cpu()*1e3, cmap="binary", shading="auto")
        plt.colorbar()
        plt.ylabel("x (um)")
        plt.xlabel("z (mm)")
        plt.ylim(-xlim, xlim)
        plt.gca().invert_xaxis()
        plt.title(r"$\Delta n\ \  (10^{-3})$")
        plt.grid(alpha=0.5)

    def _plot_Imat_x(self, xlim=200, renorm_flag=True):
        wg = self
        Emat_x = wg.Emat_x.detach().cpu().numpy()
        Imat_x = np.abs(Emat_x)**2

        if renorm_flag:
            Imat_x = (Imat_x.T/np.max(Imat_x, axis=1)).T


        plt.pcolormesh(wg.z_list.to("mm").value, wg.x_axis.to("um").value, Imat_x.T, cmap="inferno", vmin=0, shading="auto")
        plt.xlabel("z (mm)")
        plt.ylabel("x (um)")
        plt.ylim(-xlim, xlim)
        plt.gca().invert_xaxis()
        plt.title("Spatial intensity")
        plt.grid(alpha=0.5)

    def _plot_Imat_f(self, flim=80, renorm_flag=True):
        wg = self
        Emat_f = wg.Emat_f.detach().cpu().numpy()
        Imat_f = np.abs(Emat_f)**2

        if renorm_flag:
            Imat_f = (Imat_f.T/np.max(Imat_f, axis=1)).T

        plt.pcolormesh(wg.z_list.to("mm").value, wg.fx_axis.to("1/mm").value, Imat_f.T, cmap="inferno", vmin=0, shading="auto")
        plt.xlabel("z (mm)")
        plt.ylabel("f (1/mm)")
        plt.ylim(-flim, flim)
        plt.gca().invert_xaxis()
        plt.title("Wavevector intensity")
        plt.grid(alpha=0.5)

    def plot_mats(self, xlim=200, flim=80, renorm_flag=True):
        """
        renorm_flag: If true, the plots are renormalized to the maximum value of the intensity for a given zaxis point.
        """        
        wg = self #to save rewriting the code!


        fig, axs = plt.subplots(1, 3, figsize=(12, 2))
        fig.subplots_adjust(wspace=0.4)

        plt.sca(axs[0])
        self._plot_delta_n(xlim=xlim)

        plt.sca(axs[1])
        self._plot_Imat_x(xlim=xlim, renorm_flag=renorm_flag)

        plt.sca(axs[2])
        self._plot_Imat_f(flim=flim, renorm_flag=renorm_flag)

        for ax in axs:
            plt.sca(ax)

        return fig, axs

########### Utility functions outside of the main class ##########

def run_no_voltage_simulation(wg):
    delta_n_store = wg.delta_n
    wg.delta_n = torch.zeros_like(wg.delta_n)
    
    data_V = wg.run_simulation()
    wg.delta_n = delta_n_store
    return data_V

def get_gaussian_kernel(nx, nz, sigmax = 1/3, sigmaz = 1/3):
    # returns array filled with pdf of multivariate gaussian from [-1,1]x[-1,1] 
    # with nx (nz) points in first (second) dimension and standard deviation of sigmax (sigmaz)
    x, z = np.mgrid[-1:1:1j*nx, -1:1:1j*nz] # complex numbers as step size makes this a multidimensional linspace
    pos = np.dstack((x/sigmax, z/sigmaz))
    rv = multivariate_normal([0., 0.])
    weights = rv.pdf(pos)
    weights /= weights.sum()
    return weights

def calc_center_of_gravity(x_axis, beam):
    return torch.sum(torch.from_numpy(x_axis) * beam) / torch.sum(beam)

def create_curved_waveguide(wg, r, input_length, electrode_length, output_length, d_wg, carrier_diffusion_length):
    delta_n = torch.zeros(wg.Nz, wg.Nx)
    xx, zz = np.meshgrid(wg.x_axis, wg.z_axis)

    x0, z0 = r, input_length

    pattern_up = (xx-r)**2 + (zz-z0)**2 > (r-d_wg/2)**2
    pattern_down = (xx-r)**2 + (zz-z0)**2 < (r+d_wg/2)**2

    pattern = pattern_up*pattern_down

    delta_n = torch.from_numpy(pattern.astype(float))*wg.delta_n_val

    delta_n[0:wg.z2ind(input_length), :] = 0.0
    delta_n[wg.z2ind(input_length+electrode_length):wg.z2ind(wg.Lz), :] = 0.0

    delta_n = wg.smoothen_spatial_map(delta_n, carrier_diffusion_length)

    wg.delta_n = delta_n

def single_mode_width(wg):
    return wg.lam0/2/np.sqrt((wg.n+wg.delta_n_val)**2-wg.n**2)
   
from torch.nn import Upsample

def torch_resize(input, scale_factor_1, scale_factor_2):
    # scales a tensor of shape (n1, n2) to (scalscale_factor_1 * n1, scalscale_factor_2 * n2)
    upsample_1 = Upsample(scale_factor=scale_factor_1, mode='nearest') #, align_corners=True)
    upsample_2 = Upsample(scale_factor=scale_factor_2, mode='nearest') #, align_corners=True)
    
    input = input.unsqueeze(0)
    input = upsample_2(input)
    input = input.mT
    input = upsample_1(input)
    input = input.mT.squeeze(0)
    return input

def overlap_intergral(f1, f2):
    return torch.abs(torch.sum(torch.conj_physical(f1) * f2, dim = -1))**2 / torch.norm(f1, dim = -1)**2 / torch.norm(f2, dim = -1)**2

def convolve_with_circle(wg, spatial_map, radius):
    # convolves a spatial map (such as a delta_n) with a circular kernel 
    # of given radius in in um. 
    # The input should have size of [wg.Nx, wg.Nz]
    nx = int(np.round((radius/wg.dx).decompose())/2)*2+1
    nz = int(np.round((radius/wg.dz).decompose())/2)*2+1

    # create circular kernel to average surrounding pixels
    kernel = torch.FloatTensor(
        get_circular_kernel(nz, nx)
    ).unsqueeze(0).unsqueeze(0)
    # pad delta n and convolve it with gaussian kernel
    spatial_map = fft_conv(
        spatial_map.unsqueeze(0).unsqueeze(0), 
        kernel, bias=None, padding = [int(0.5*nz),int(0.5*nx)], 
        padding_mode='reflect')
    return spatial_map.squeeze(0).squeeze(0)

def get_circular_kernel(nx, nz, radius=1):
    # returns a circular kernel of dimension [nx, nz]
    # normalized such that kernel.sum() = 1
    x, z = np.mgrid[-1:1:1j*nx, -1:1:1j*nz] # complex numbers as step size makes this a multidimensional linspace
    pos = np.dstack((x/radius, z/radius))
    dist = np.sqrt((pos**2).sum(axis = -1))
    kernel = (dist<1).astype(float)
    kernel /= kernel.sum()
    return kernel
