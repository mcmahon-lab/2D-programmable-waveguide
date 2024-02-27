from collections import OrderedDict
import numpy as np
import csv
import os
import scipy.constants as const
from tdwg.lib.electrical_helpers import parallel_plate_cap, resistance2resistivity, resistivity2resistance, V2E, E2V
from tdwg.lib.modified_nodal_analysis import modified_nodal_analysis
from tdwg.lib.optical_helpers import index2theta_in_air_to_chip, theta_in_air_to_chip2index
import matplotlib.pyplot as plt

ZERO = np.finfo(float).eps
ALL_DEVICES = []

class Layer():
    def __init__(
            self, 
            material = None, 
            thickness = None, 
            n = None, 
            area = None, 
            gas_ratio = None,
            breakdown = None,
            resistance = None,
            capacitance = None,
            eps_r = 1, # dielectric constant,
            comment = ''
            ):
        self.thickness = thickness # m
        self.n = n # complex refractive index
        self.area = area # m^2
        self.gas_ratio = gas_ratio # SiH4/NH3
        self.breakdown = breakdown # electric breakdown field (V/um)
        self.eps_r = eps_r
        self.comment = comment
        
        if material == 'SRN':
            self.resistance = self.get_resistance_func()
            self.bright_resistance = self.get_resistance_func(dark = False)
        if isinstance(resistance, float):
            self.resistance = lambda V: resistance
        if area and thickness:
            self.parallel_plate_cap_from_geometry()
        else: 
            self.capacitance = capacitance
            
        
    def parallel_plate_cap_from_geometry(self):
        self.capacitance = parallel_plate_cap(A=self.area, d=self.thickness, eps_r=self.eps_r)
    
    def thickness_from_capacitance(self):
        self.thickness = const.epsilon_0 * self.eps_r * self.area / self.capacitance
        
    def get_resistance_func(self, dark = True):
        def resistance_(V):
            E = V2E(V, self.thickness)
            resistivity = poole_frenkel_resistivity(E, gas_ratio = self.gas_ratio, dark = dark)
            return resistivity2resistance(resistivity, A = self.area, d = self.thickness)
        
        return resistance_
        
        
class Mode():
    def __init__(
            self, 
            neff = None,
            loss = None, #db/cm
            lambda0 = None,
            polarization = None,
            ):
        self.neff = neff
        self.loss = loss
        self.lambda0 = lambda0
        self.polarization = polarization
        
    def __str__(self):
        return f'{self.polarization} mode at neff = {self.neff} @ {self.lambda0*1e9}nm'
    
    
class Device(OrderedDict):
    def __init__(self, name, substrate, nickname = None):
        super().__init__()
        self.modes = []
        self.name = name 
        self.nickname = nickname 
        self.substrate = substrate
        ALL_DEVICES.append(self)
        
    def add_layer(self, name, **kwargs):
        self[name] = Layer(**kwargs)
        
    def add_mode(self, **kwargs):
        self.modes.append(Mode(**kwargs))
        
    def get_prism_coupling_angles(self, prism):
        for mode in self.modes:
            neff = mode.neff
            
            if mode.polarization == 'TE':
                nprism = prism.n_e
            if mode.polarization == 'TM':
                nprism = prism.n_o
                
            angle = np.rad2deg(index2theta_in_air_to_chip(neff, nprism))
            print(mode)
            print(prism)
            print(f'Prism coupling angle {angle:.2f}deg \n')
    
    def write_to_csv(self):
       # get first layer...
        layer0 = next(iter(self))
        # ...to get a list of all columns in the excel sheet
        fieldnames = ['name'] + list(vars(self[layer0]).keys())
        
        if not os.path.exists('device_properties'):
            os.mkdir('device_properties')
        with open('device_properties/' + self.name + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # write all properties for all layers
            for layer, properties in self.items():
                propertydict = vars(properties)
                propertydict['name'] = layer
                writer.writerow(propertydict)
    

class Circuit():
    def __init__(self, device):
        self.device = device
    
    def create_voltage_source(self, V0, f, dt = None):
        self.Vd = lambda t: V0 * np.sin(2 * np.pi * f * t)
        if not dt:
            self.dt = 1/f/1000
        else:
            self.dt = dt
        
    def simulate(self, t0, tmax, dt = None):
        if dt:
            self.dt = dt
        
        Cs, self.R_dark_funcs = self.build_RC_list(dark = True)
        Cs, self.R_bright_funcs = self.build_RC_list(dark = False)
        
        self.t_bright, self.V_bright, self.IR_bright, self.IC_bright, self.R_bright = modified_nodal_analysis(t0, self.dt, tmax, self.R_bright_funcs, Cs, self.Vd)
        self.t_dark, self.V_dark, self.IR_dark, self.IC_dark, self.R_dark = modified_nodal_analysis(t0, self.dt, tmax, self.R_dark_funcs, Cs, self.Vd)
        
        self.I_dark = self.IR_dark.sum(axis=-1) + self.IC_dark.sum(axis=-1)
        self.I_bright = self.IR_bright.sum(axis=-1) + self.IC_bright.sum(axis=-1)
        
        Vdrop_dark = -np.diff(self.V_dark, axis = -1)
        Vdrop_bright = -np.diff(self.V_bright, axis = -1)
        self.Vdrop_dark = dict()
        self.Vdrop_bright = dict()
        self.Emax_dark = dict()
        self.Emax_bright = dict()
        for i, name in enumerate(self.names):
            self.Vdrop_dark[name] = Vdrop_dark[:,i]
            self.Vdrop_bright[name] = Vdrop_bright[:,i]
            self.Emax_dark[name] = Vdrop_dark[:,i].max() / self.device[name].thickness
            self.Emax_bright[name] = Vdrop_bright[:,i].max() / self.device[name].thickness
        
        return self.t_bright, self.V_dark, self.V_bright, self.I_dark, self.I_bright

        
    def build_RC_list(self, dark = True):
        Cs, R_funcs, self.names = [], [], []
        
        for name, layer in self.device.items():
            # skip electrodes to avoid numerical instabilities due to low resistances
            if 'electrode' in name:
                continue
            # create list of layer names
            self.names.append(name)
            Cs.append(layer.capacitance)
            if name == 'core':
                if dark:
                    R_funcs.append(layer.resistance)
                else:
                    R_funcs.append(layer.bright_resistance)
            else:
                R_funcs.append(layer.resistance)
                
        return np.array(Cs), np.array(R_funcs)
    
    def calc_power_dissipation(self):
        self.P_bright = self.IR_bright**2 * self.R_bright
        self.P_dark = self.IR_dark**2 * self.R_dark
        #self.P_bright = np.zeros_like(self.IR_bright)
        #self.P_dark = np.zeros_like(self.IR_dark)
        
        #for i in range(len(self.names)):
            #self.P_bright[:,i] = self.IR_bright[:,i]**2 / self.R_bright_funcs[i](np.diff(self.V_bright)[:,i])
            #self.P_dark[:,i] = self.IR_dark[:,i]**2 / self.R_dark_funcs[i](np.diff(self.V_dark)[:,i])
            #self.P_dark[:,i] = np.diff(self.V_dark)[:,i]**2/self.R_dark_funcs[i](np.diff(self.V_dark)[:,i])
            #self.P_bright[:,i] = np.diff(self.V_bright)[:,i]**2/self.R_bright_funcs[i](np.diff(self.V_bright)[:,i])
            
    #def find_induced_delta_n(self):
    
    def calc_delta_n(self, n0, chi3):
        core_idx = np.argmax([name == 'core' for name in self.names])
        Vdrop_dark = -np.diff(self.V_dark, axis = -1)
        Vdrop_bright = -np.diff(self.V_bright, axis = -1)
        Emax_dark = Vdrop_dark[:, core_idx].max() / self.device['core'].thickness
        Emax_bright = Vdrop_bright[:, core_idx].max() / self.device['core'].thickness

        delta_eps_dark = 12 * chi3 * Emax_dark**2
        delta_eps_bright = 12 * chi3 * Emax_bright**2

        n_dark = np.sqrt(n0**2 + delta_eps_dark)
        n_bright = np.sqrt(n0**2 + delta_eps_bright)

        return n_dark, n_bright
        
    
    def plot_voltage_per_component(self, dark = True): 
        for i in range(len(self.names)):
            label = self.names[i]
            if dark:
                plt.plot(self.t_dark, self.V_dark[:,i] - self.V_dark[:,i+1], label = label)
            else:
                plt.plot(self.t_bright, self.V_bright[:,i] - self.V_bright[:,i+1], label = label)
        plt.title(f"Voltage across each layer in {'dark' if dark else 'bright'} state")
        plt.ylabel('V')
        plt.xlabel('time')
        plt.legend()
        plt.show()
        
    def plot_dark_v_bright_current(self):
        plt.title(f"Current across device in dark and bright state")
        plt.plot(self.t_dark, self.I_dark, label = 'dark current')
        plt.plot(self.t_bright, self.I_bright, label = 'bright current')
        plt.ylabel('I (A)')
        plt.xlabel('time (s)')
        plt.legend()
                
def poole_frenkel_resistance(V, a, b, c, d = 1e-7):
    V = np.abs(V)
    sigma = a * V * np.exp(b*np.sqrt(V) - c) + d
    return 1/sigma


def poole_frenkel_resistivity(E, gas_ratio, dark = True):
    if gas_ratio == W3Jan25['core'].gas_ratio:
        V = E2V(E, W3Jan25['core'].thickness)
        if dark:
            R = poole_frenkel_resistance(V, 7e-15, 2.5, 0)
        if not dark:
            R = poole_frenkel_resistance(V, 1.1e-6, 0.185, 0)
        rho = resistance2resistivity(R, A = W3Jan25['core'].area, d = W3Jan25['core'].thickness)
    else:
        print(f"No resistivity data for gas ratio {gas_ratio}")
        return None    
    return rho


def n2_to_chi3(n2, n0):
    return 4/3 * n2 * n0**2 * const.epsilon_0 * const.c
    
        
#########################################################
W3Jan25 = Device('W3Jan25', nickname = '50/10 SRN-only chip', substrate = 'SiO2')
W3Jan25.add_layer('top-electrode', material = 'Au', thickness = 15e-9, resistance = ZERO,
                 area = 0.64e-4 # measured by ruler for undamaged electrode slighlty off-center
                 )
W3Jan25.add_layer('core', material = 'SRN', gas_ratio = 50./10.,
                 comment = 'Nominally 300nm thick, realistically much thinner.',
                 capacitance = 53e-9, # measured by current measurement at 1kHz 
                 eps_r = 7, # assuming eps_r of SiN from http://www.mit.edu/~6.777/matprops/pecvd_sin.htm
                 area = 0.64e-4 # measured by ruler for undamaged electrode slighlty off-center
                 )
W3Jan25.add_layer('bottom-electrode', material = 'ITO', thickness = 15e-9, resistance = ZERO)

W3Jan25['core'].thickness_from_capacitance()

#########################################################
W4Jan25 = Device('W4Jan25', nickname = '50/5 SRN-only chip', substrate = 'SiO2')
W4Jan25.add_layer('top-electrode', material = 'Au', thickness = 15e-9, resistance = ZERO)
W4Jan25.add_layer('core', material = 'SRN', gas_ratio = 50./20.,
            comment = 'Nominally 600nm thick, realistically much thinner.')
W4Jan25.add_layer('bottom-electrode', material = 'ITO', thickness = 15e-9, resistance = ZERO)


#########################################################
W2Mar15 = Device('W2Mar15', nickname = 'first SRN waveguide', substrate = 'SiO2')
W2Mar15.add_layer('top-electrode', material = 'Au', thickness = 15e-9, resistance = ZERO)
W2Mar15.add_layer('top-cladding', material = 'SiO2', thickness = 1e-6, resistance = np.inf)
W2Mar15.add_layer('core', material = 'SRN', comment = 'thickness varies, at the thinnest point very <100nm')
W2Mar15.add_layer('bottom-cladding', material = 'SiO2', thickness = 1e-6, resistance = np.inf)


#########################################################
W3Mar15 = Device('W3Mar15', nickname = 'first SiN waveguide', substrate = 'SiO2')
W3Mar15.add_layer('top-cladding', material = 'SiO2', thickness = 1e-6, resistance = np.inf)
W3Mar15.add_layer('core', material = 'SiN', thickness = 0.45e-6)
W3Mar15.add_layer('bottom-cladding', material = 'SiO2', thickness = 1e-6, resistance = np.inf)


#########################################################
W6 = Device('W6', substrate = 'SiO2')
W6.add_layer('top-electrode', material = 'Au', thickness = 15e-9, 
             area = 1.48*1.01*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
             resistance = ZERO)
W6.add_layer('top-cladding', material = 'SiO2', resistance = np.inf,
            area = 1.48*1.01*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            thickness = 1.12e-6, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            eps_r = 3.9, # https://www.iue.tuwien.ac.at/phd/filipovic/node26.html
            )
W6.add_layer('core', material = 'SRN', 
            thickness = 620e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            gas_ratio = 50./10., # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625943425184100
            area = 1.48*1.01*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            eps_r = 3.9, # https://www.iue.tuwien.ac.at/phd/filipovic/node26.html
            )
W6.add_layer('bottom-cladding', material = 'SiO2', resistance = np.inf,
            area = 1.48*1.01*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            thickness = 1.12e-6, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            eps_r = 3.9, # https://www.iue.tuwien.ac.at/phd/filipovic/node26.html
            )
W6.add_layer('bottom-electrode', material = 'ITO', thickness = 15e-9, resistance = ZERO,
             area = 1.48*1.01*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
            )


#########################################################
W8 = Device('W8', substrate = 'SiO2')
W8.add_layer('top-electrode', material = 'Au', thickness = 15e-9, area = 1.48*1.01*1e-4, resistance = ZERO)
W8.add_layer('core', material = 'SiO2', resistance = np.inf)
W8.add_layer('bottom-electrode', material = 'ITO', thickness = 15e-9, resistance = ZERO)


#########################################################
W9 = Device('W9', substrate = 'SiO2')
W9.add_layer('top-electrode', material = 'Au', thickness = 15e-9, area = 1.48*1.01*1e-4, resistance = ZERO)
W9.add_layer('core', material = 'SiO2', resistance = np.inf)
W9.add_layer('bottom-electrode', material = 'ITO', thickness = 15e-9, resistance = ZERO)


#########################################################
W10 = Device('W10', substrate = 'SiO2')
W10.add_layer('top-electrode', material = 'Au', thickness = 15e-9, area = 1.06*0.89*1e-4, resistance = ZERO)
W10.add_layer('top-cladding', material = 'SiO2', resistance = np.inf,
              thickness = 1.15e-6, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
              area = 1.06*0.89*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
             )
W10.add_layer('core', material = 'SRN', 
              thickness = 700e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
              gas_ratio = 50./5., # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625943425184100
              area = 1.06*0.89*1e-4 # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
             )
W10.add_layer('bottom-cladding', material = 'SiO2', resistance = np.inf,
              thickness = 1.15e-6, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
              area = 1.06*0.89*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
             )
W10.add_layer('bottom-electrode', material = 'ITO', thickness = 15e-9, resistance = ZERO,
              area = 1.06*0.89*1e-4, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1625872722173100
             )


#########################################################
WB1 = Device('WB1', nickname = 'SIN on glass prism coupler', substrate = 'SiO2')
WB1.add_layer('core', material = 'SiN', 
              thickness = 500e-9, # nominally (https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1628043241032900?thread_ts=1627686704.028100&cid=G01614TSBM5)
             )
WB1.add_layer('bottom-cladding', material = 'SiO2', resistance = np.inf,
             thickness = 1.5e-6 # nominally (https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1628043241032900?thread_ts=1627686704.028100&cid=G01614TSBM5)
             )

WB1.add_mode(neff = 1.85, lambda0 = 637e-9, loss = 13.12, polarization = 'TE')
WB1.add_mode(neff = 1.70, lambda0 = 637e-9, polarization = 'TE')
WB1.add_mode(neff = 1.63, lambda0 = 637e-9, polarization = 'TE')
WB1.add_mode(neff = 1.753, lambda0 = 1059e-9, loss = 13.09, polarization = 'TE')
WB1.add_mode(neff = 1.438, lambda0 = 1059e-9, polarization = 'TE')
WB1.add_mode(neff = 1.509, lambda0 = 1059e-9, polarization = 'TE')
WB1.add_mode(neff = 1.656, lambda0 = 1549e-9, loss = 17.66, polarization = 'TE')
WB1.add_mode(neff = 1.39, lambda0 = 1549e-9, polarization = 'TE')


#########################################################
WB2 = Device('WB2', nickname = 'SRN on glass prism coupler', substrate = 'SiO2')
WB2.add_layer('core', material = 'SRN', 
              thickness = 500e-9, # nominally (https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1628043241032900?thread_ts=1627686704.028100&cid=G01614TSBM5)
              gas_ratio = 50./5., # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1627686704028100
             )
WB2.add_layer('bottom-cladding', material = 'SiO2', resistance = np.inf,
             thickness = 1.5e-6 # nominally (https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1628043241032900?thread_ts=1627686704.028100&cid=G01614TSBM5)
             )

WB2.add_mode(neff = 1.61, lambda0 = 1549e-9, loss = 28.55, polarization = 'TE')
WB2.add_mode(neff = 1.39, lambda0 = 1549e-9, polarization = 'TE')


#########################################################
WB3 = Device('WB3', nickname = 'SiN on Si prism coupler', substrate = 'Si')
WB3.add_layer('core', material = 'SiN', 
              thickness = 500e-9, # nominally (https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1628043241032900?thread_ts=1627686704.028100&cid=G01614TSBM5)
             )
WB3.add_layer('bottom-cladding', material = 'SiO2', resistance = np.inf,
             thickness = 1.5e-6 # nominally (https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1628043241032900?thread_ts=1627686704.028100&cid=G01614TSBM5)
             )

WB3.add_mode(neff = 1.81, lambda0 = 637e-9, polarization = 'TE')
WB3.add_mode(neff = 1.50, lambda0 = 637e-9, polarization = 'TE')
WB3.add_mode(neff = 1.58, lambda0 = 1549e-9, loss = 2.19, polarization = 'TE')
WB3.add_mode(neff = 1.689, lambda0 = 1064e-9, loss = 12.99, polarization = 'TE')
WB3.add_mode(neff = 1.423, lambda0 = 1064e-9, polarization = 'TE')


#########################################################
WSi_alpha4 = Device('WSi_alpha4', substrate = 'Si')
WSi_alpha4.add_layer('core', gas_ratio = 4)
WSi_alpha4.add_mode(neff = 2.035, lambda0 = 1549e-9, loss = 2.45, polarization = 'TE')
WSi_alpha4.add_mode(neff = 1.713, lambda0 = 1549e-9, polarization = 'TE')
WSi_alpha4.add_mode(neff = 1.419, lambda0 = 1549e-9, polarization = 'TE')
WSi_alpha4.add_mode(neff = 1.942, lambda0 = 824e-9, polarization = 'TE')
WSi_alpha4.add_mode(neff = 2.157, lambda0 = 1064-9, polarization = 'TE')
WSi_alpha4.add_mode(neff = 1.73, lambda0 = 1064-9, polarization = 'TE')
WSi_alpha4.add_mode(neff = 1.421, lambda0 = 1064-9, polarization = 'TE')

    
#########################################################
WSi_alpha2 = Device('WSi_alpha4', substrate = 'Si')
WSi_alpha2.add_layer('core', gas_ratio = 2)
WSi_alpha2.add_mode(neff = 1.723, lambda0 = 1549e-9, polarization = 'TE')
WSi_alpha2.add_mode(neff = 1.93, lambda0 = 824e-9, polarization = 'TE')
WSi_alpha2.add_mode(neff = 1.464, lambda0 = 824e-9, polarization = 'TE')

    
#########################################################
WSiO2_alpha3 = Device('WSi_alpha4', substrate = 'SiO2')
WSiO2_alpha3.add_layer('core', gas_ratio = 3)
WSiO2_alpha3.add_mode(neff = 1.9, lambda0 = 1549e-9, loss = 3.5, polarization = 'TE')
WSiO2_alpha3.add_mode(neff = 2.215, lambda0 = 637-9, polarization = 'TE')
WSiO2_alpha3.add_mode(neff = 1.956, lambda0 = 637-9, polarization = 'TE')
WSiO2_alpha3.add_mode(neff = 1.52, lambda0 = 637-9, polarization = 'TE')

    
#########################################################
WSiO2_alpha15 = Device('WSi_alpha15', substrate = 'SiO2')
WSiO2_alpha15.add_layer('core', gas_ratio = 1.5)
WSiO2_alpha15.add_mode(neff = 1.615, lambda0 = 1549e-9, polarization = 'TE')


P4 = Device('P4', substrate = 'Si')
P4.add_layer('core', material = 'SRN', gas_ratio = 4,
            thickness = 476e-9, # fit from prism coupling modes
            n = 2.276, # fit from prism coupling modes
            )
P4.add_layer('bottom-cladding', material = 'SiO2',
            thickness = 1500e-9, # nominal, https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1636993908085100
            eps_r = 3.9, # https://en.wikipedia.org/wiki/Relative_permittivity
            n = 1.444,
            )

P6 = Device('P6', substrate = 'Si')
P6.add_layer('core', material = 'SRN', gas_ratio = 6,
            thickness = 486e-9, # fit from prism coupling modes
            n = 2.338, # fit from prism coupling modes
            )
P6.add_layer('bottom-cladding', material = 'SiO2',
            thickness = 1500e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1636993908085100,
            eps_r = 3.9, # https://en.wikipedia.org/wiki/Relative_permittivity
            n = 1.444,
            )

P10 = Device('P10', substrate = 'Si')
P10.add_layer('core', material = 'SRN', gas_ratio = 10,
            thickness = 486, # fit from prism coupling modes
            n = 2.476, # fit from prism coupling modes
            )
P10.add_layer('bottom-cladding', material = 'SiO2',
            thickness = 1500e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1636993908085100
            eps_r = 3.9, # https://en.wikipedia.org/wiki/Relative_permittivity
            n = 1.444,
            )

P4.add_mode(neff = 1.85, lambda0 = 1549e-9, loss = 6.39, polarization = 'TM') # ms3452\Jointly fabricated chips\21-11-12 MB2 P4\1550 loss 11-29 TM.xps
P4.add_mode(neff = 2.03, lambda0 = 1549e-9, loss = 5.47, polarization = 'TE') # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1637118163086400?thread_ts=1636993908.085100&cid=G01614TSBM5 
P4.add_mode(neff = 1.72, lambda0 = 1549e-9, polarization = 'TE') # ms3452\Jointly fabricated chips\21-11-12 MB2 P4\1550 modes 11-29.xps
P4.add_mode(neff = 1.41, lambda0 = 1549e-9, polarization = 'TE') # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1637118163086400?thread_ts=1636993908.085100&cid=G01614TSBM5
P4.add_mode(neff = 2.14, lambda0 = 633e-9, polarization = 'TE') # ms3452\Jointly fabricated chips\21-11-12 MB2 P4\633 modes.xps
P4.add_mode(neff = 1.79, lambda0 = 633e-9, polarization = 'TE') # ms3452\Jointly fabricated chips\21-11-12 MB2 P4\633 modes.xps

P6.add_mode(neff = 2.10, lambda0 = 1549e-9, polarization = 'TE') # ms3452\Jointly fabricated chips\21-11-12 TOMS1 P6/1550 modes TE.xps
P6.add_mode(neff = 1.43, lambda0 = 1549e-9, polarization = 'TE') # ms3452\Jointly fabricated chips\21-11-12 TOMS1 P6/1550 modes TE.xps
P6.add_mode(neff = 1.92, lambda0 = 1549e-9, loss = 5.58, polarization = 'TM') # ms3452\Jointly fabricated chips\21-11-12 TOMS1 P6/1550 modes TM.xps
P6.add_mode(neff = 1.887, lambda0 = 1059e-9, polarization = 'TM') # Martin's personal notes, 2022/01/27

P10.add_mode(neff = 2.24, lambda0 = 1549e-9, loss = 2.72, polarization = 'TE') # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1637118163086400?thread_ts=1636993908.085100&cid=G01614TSBM5
P10.add_mode(neff = 1.53, lambda0 = 1549e-9, polarization = 'TE') # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1637118163086400?thread_ts=1636993908.085100&cid=G01614TSBM5
P10.add_mode(neff = 2.06, lambda0 = 1549e-9, loss = 7.04, polarization = 'TM') # ms3452\Jointly fabricated chips\21-11-12 MB3 P10/1550 modes 11-29 TM.xps


#########################################################
SRN4_MSTO_22_01_04 = Device('22-01-04 MSTO SRN_4', substrate = 'Si')
SRN4_MSTO_22_01_04.add_layer('core', material = 'SRN', gas_ratio = 4,
            thickness = 500e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1641530368008500
            area = 12.5e-3*14e-3, # innermost electrode (at the 90deg corner)
            )

SRN6_MSTO_22_01_04 = Device('22-01-04 MSTO SRN_6', substrate = 'Si')
SRN6_MSTO_22_01_04.add_layer('core', material = 'SRN', gas_ratio = 6,
            thickness = 500e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1641530368008500
            area = 12e-3*12e-3, # innermost electrode (at the 90deg corner)
            )

SRN8_MSTO_22_01_04 = Device('22-01-04 MSTO SRN_8', substrate = 'Si')
SRN8_MSTO_22_01_04.add_layer('core', material = 'SRN', gas_ratio = 8,
            thickness = 500e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1641530368008500
            area = 12e-3*12e-3, # innermost electrode (at the 90deg corner)
            )

SRN10_MSTO_22_01_04 = Device('22-01-04 MSTO SRN_10', substrate = 'Si')
SRN10_MSTO_22_01_04.add_layer('core', material = 'SRN', gas_ratio = 10,
            thickness = 500e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1641530368008500
            area = 11e-3*14e-3, # innermost electrode (at the 90deg corner)
            )

##########################################################
SRN3_TO_22_02_01 = Device('22-02-01 TO SRN_3', substrate = 'Si')
SRN3_TO_22_02_01.add_layer('top-cladding', material = 'SiO2',
            thickness = 1000e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1644267545891919
            eps_r = 3.9, # https://en.wikipedia.org/wiki/Relative_permittivity
            n = 1.444,
            )
SRN3_TO_22_02_01.add_layer('core', material = 'SRN', gas_ratio = 3,
            thickness = 600e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1644267545891919
            area = 1e-4 # not measured, just a rough estimate
            )
SRN3_TO_22_02_01.add_layer('bottom-cladding', material = 'SiO2',
            thickness = 1000e-9, # https://mcmahon-lab.slack.com/archives/G01614TSBM5/p1644267545891919
            eps_r = 3.9, # https://en.wikipedia.org/wiki/Relative_permittivity
            n = 1.444,
            )