from tdwg.lib.electrical_helpers import imp_series, imp_parallel, imp_cap, parallel_plate_cap, cap_series, cap_parallel
from tdwg.lib.optical_helpers import r_eff_to_delta_n, modulation_efficiency_TM
from tdwg.lib.modified_nodal_analysis import modified_nodal_analysis
import numpy as np
import types 
import copy
    
class WG_base():
    """
    Base class for stack of photoconductor, top-cladding, E-O core, bottom-cladding
    all variables in base SI units
    """
    def __init__(
        self,
        # geometrical properties 
        A = 8e-3 * 8e-3, # electrode area (m^2)
        d_pc = 470e-9, # photoconductor thickness (m)
        d_tcl = 1.0e-6, # top cladding thickness (m)
        d_co = 600e-9, # core thickness (m)
        d_bcl = 2e-6, # bottom cladding thickness (m)
        #
        # optical properties
        r33 = 25e-12, # m/V
        n_co = 2.2,
        n_cl = 1.444,
        lambda0 = 1.55e-6,
    ):
        self.A = A # electrode area (m^2)
        self.d_pc = d_pc # photoconductor thickness (m)
        self.d_tcl = d_tcl # top cladding thickness (m)
        self.d_co = d_co # core thickness (m)
        self.d_bcl = d_bcl # bottom cladding thickness (m)
        self.d_cl = d_tcl + d_bcl # cladding thickness (m)
        
        self.r33 = r33 # core Pockels coefficient (m/V)
        self.n_co = n_co # core index
        self.n_cl = n_cl
        self.lambda0 = lambda0
        self.modulation_efficiency_TM()
        
    def print_device(self, f, V):
        print_horizontal_wire()
        print_voltage_source(f, V)
        for (R, C) in zip(self.Rs, self.Cs):
            print_layer(R, C, f, ohm_per_line=0.3e6)
        print_horizontal_wire()
    
    def modulation_efficiency_TM(self, dn = 1e-3, plot = False):
        self.modulation_efficiency = modulation_efficiency_TM(self, lambda0 = self.lambda0, dn = dn, plot = plot)
        return self.modulation_efficiency

class WG_linear(WG_base):
    """
    Linear electrical stack of photoconductor, top-cladding, E-O core, bottom-cladding
    all variables in base SI units
    """
    def __init__(
        self,
        # electrical permittivities
        eps_pc = 6,
        eps_cl = 3.9,
        eps_co = 27,
        #
        # electrical resistivities
        rho_pc = 2.86667574e+06 * 7e-3 * 8e-3 / 600e-9,
        rho_cl = np.inf,
        rho_co = np.inf,
        R_substrate = 0, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.eps_pc = eps_pc # photoconductor dielectric permittivity
        self.eps_cl = eps_cl # cladding dielectric permittivity
        self.eps_co = eps_co # core dielectric permittivity
        
        self.rho_pc = rho_pc # photoconductor resistivity (Ohm.m)
        self.rho_cl = rho_cl # core resistivity (Ohm.m)
        self.rho_co = rho_co # core resistivity (Ohm.m)
        
        self.R_pc = self.rho_pc * self.d_pc / self.A # photoconductor resistance (Ohm)
        self.R_tcl = self.rho_cl * self.d_tcl / self.A # top cladding resistance (Ohm)
        self.R_co = self.rho_co * self.d_co / self.A # core resistance (Ohm)
        self.R_bcl = self.rho_cl * self.d_bcl / self.A # bottom cladding resistance (Ohm)
        self.R_substrate = R_substrate # substrate resistance (Ohm)
        
        self.C_pc = parallel_plate_cap(self.A, self.d_pc, self.eps_pc)
        self.C_tcl = parallel_plate_cap(self.A, self.d_tcl, self.eps_cl)
        self.C_co = parallel_plate_cap(self.A, self.d_co, self.eps_co)
        self.C_bcl = parallel_plate_cap(self.A, self.d_bcl, self.eps_cl)
        self.C_cl = parallel_plate_cap(self.A, self.d_cl, self.eps_cl)
        
        self.Cs = np.array([self.C_pc, self.C_tcl, self.C_co, self.C_bcl])
        self.Rs = np.array([self.R_pc, self.R_tcl, self.R_co, self.R_bcl])
        
    def imp_core(self, f):
        return imp_parallel([imp_cap(f, self.C_co), self.R_co])
    
    def imp_wg(self, f):
        return imp_series([self.imp_core(f), imp_cap(f, self.C_cl)])
        
    def imp_pc(self, f):
        return imp_parallel([imp_cap(f, self.C_pc), self.R_pc])

    def imp_total(self, f):
        return imp_series([self.R_substrate, self.imp_wg(f), self.imp_pc(f)])
    
    def Z_ratio(self, f):
        return np.abs(self.imp_core(f) / self.imp_total(f))
    
    def Z_ratio_pc(self, f):
        return np.abs(self.imp_pc(f) / self.imp_total(f))
    
    def E_pc(self, f, Vapplied):
        return self.Z_ratio_pc(f) * Vapplied / self.d_pc
    
    def E_co(self, f, Vapplied):
        return self.Z_ratio(f) * Vapplied / self.d_co

    def delta_n(self, f, Vapplied): 
        return 1/2 * self.n_co**3 * self.r33 * self.E_co(f, Vapplied)
    
    def I_total(self, f, Vapplied):
        return np.abs(Vapplied / self.imp_total(f))
    
    def delta_n_core(self, f, Vapplied):
        Z_ratio_core = self.Z_ratio(f)
        Z_ratio_pc = self.Z_ratio_pc(f)
        E_co_max = self.E_co(f, Vapplied)
        E_pc_max = self.E_pc(f, Vapplied)
        return r_eff_to_delta_n(self.r33, self.n_co, E_co_max)
    
    def delta_n_eff(self, f, Vapplied):
        dn_core = self.delta_n_core(f = f, Vapplied = Vapplied)
        return self.modulation_efficiency * dn_core
    
    
class WG_nonlin(WG_base):
    """
    Nonlinear electrical stack of photoconductor, top-cladding, E-O core, bottom-cladding
    The nonlinearity is in the dependence of the conductivity of all layers on the applied voltage
    all variables in base SI units
    """
    def __init__(
        self, 
        # electrical permittivities
        eps_pc = 6,
        eps_cl = 3.9,
        eps_co = 27,
        #
        # electrical resistivities
        rho_pc = lambda E: np.inf,
        rho_cl = lambda E : np.inf,
        rho_co = lambda E: np.inf,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.eps_pc = eps_pc # photoconductor dielectric permittivity
        self.eps_cl = eps_cl # cladding dielectric permittivity
        self.eps_co = eps_co # core dielectric permittivity
        
        self.rho_pc = rho_pc # photoconductor resistivity (Ohm.m)
        self.rho_cl = rho_cl # core resistivity (Ohm.m)
        self.rho_co = rho_co # core resistivity (Ohm.m)
        
        self.R_pc= lambda E : self.rho_pc(E) * self.d_pc / self.A # photoconductor resistance (Ohm)
        self.R_tcl= lambda E : self.rho_cl(E) * self.d_tcl / self.A # photoconductor resistance (Ohm)
        self.R_co = lambda E : self.rho_co(E) * self.d_co / self.A # core resistance (Ohm)
        self.R_bcl= lambda E : self.rho_cl(E) * self.d_bcl / self.A # photoconductor resistance (Ohm)
        
        self.C_pc = parallel_plate_cap(self.A, self.d_pc, self.eps_pc)
        self.C_tcl = parallel_plate_cap(self.A, self.d_tcl, self.eps_cl)
        self.C_co = parallel_plate_cap(self.A, self.d_co, self.eps_co)
        self.C_bcl = parallel_plate_cap(self.A, self.d_bcl, self.eps_cl)
        self.C_cl = parallel_plate_cap(self.A, self.d_cl, self.eps_cl)
        
        self.Cs = np.array([self.C_pc, self.C_tcl, self.C_co, self.C_bcl])
        self.R_funcs= [
            lambda V : self.R_pc(V/self.d_pc),
            lambda V : self.R_tcl(V/self.d_tcl),
            lambda V : self.R_co(V/self.d_co),
            lambda V : self.R_bcl(V/self.d_bcl),
        ]
        
    def modified_nodal_analysis(self, f, Vapplied, ncycles=100, npercycle=100):
        Vd = lambda t : Vapplied * np.sin(2*np.pi*f*t)
        t0 = 0
        tmax = ncycles/f
        dt = 1/f/npercycle
        self.t, self.Vs, self.IRs, self.ICs, self.Rs = modified_nodal_analysis(0, dt, tmax, self.R_funcs, self.Cs, Vd)
    
    def delta_n_core(self, f, Vapplied):
        self.modified_nodal_analysis(f, Vapplied)

        # find field in core
        Vdiff = np.diff(self.Vs)
        E_co = Vdiff[:,2] / self.d_co
        E_pc = Vdiff[:,0] / self.d_pc

        # find steady state voltages (by looking at end of simulation)
        E_co_max = (E_co[8000:].max() - E_co[8000:].min()) / 2
        E_pc_max = (E_pc[8000:].max() - E_pc[8000:].min()) / 2

        return r_eff_to_delta_n(self.r33, self.n_co, E_co_max), copy.deepcopy(self)
    
    def delta_n_eff(self, f, Vapplied):
        dn_core = self.delta_n_core(f = f, Vapplied = Vapplied)[0]
        return self.modulation_efficiency * dn_core, self

            
def print_horizontal_wire():
    print('    o-----------o    ')

def print_voltage_source(f, V):
    print('    |           |   ')
    print('   ---          |   ')
    print(' /     \        |   ' + f'    V = {V:.1f} V')
    print('|  /\/  |       |   ' + f'    f = {f:.1f} Hz')
    print(' \     /        |   ')
    print('   ---          |   ')


def print_layer(R, C, f, ohm_per_line = 1e6):
    Z = np.abs(imp_parallel([imp_cap(f, C), R]))
    Z_string = ['|'] * int(np.round(Z/ohm_per_line))
    separator = ""
    joined_string = separator.join(Z_string)
    print('    |        o--o--o')
    print('    |        |     \\')
    print('    |       ---    /' + f'    C = {C*1e9:.1f} nF')
    print('    |       ---    \\' + f'    R = {R*1e-6:.1f} MOhm')
    print('    |        |     /' + f'   |Z|= {Z*1e-6:.1f} MOhm \t' + joined_string)
    print('    |        o--o--o')
    print('    |           |   ')