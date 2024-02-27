import numpy as np
import matplotlib.pyplot as plt

def update_R(R_funcs, V):
    Rs = np.zeros(len(R_funcs))
    for i in range(len(Rs)):
        Rs[i] = R_funcs[i](V[i]-V[i+1])
    return Rs

def create_2D_device_G_matrix(Rs, RCeqs):
    G = np.zeros([len(Rs)+1, len(Rs)+1])
    G[0,0] = 1
    G[-1,-1] = 1
    for i in range(1,len(Rs)):
        G[i, i-1] = 1/Rs[i-1] + 1/RCeqs[i-1]
        G[i, i] = -1/Rs[i-1] - 1/RCeqs[i-1] - 1/Rs[i] - 1/RCeqs[i]
        G[i, i+1] = 1/Rs[i] + 1/RCeqs[i]
    return G

def create_2D_device_current_sources(Cs, V, Vd, dt):
    I = np.zeros(len(V))
    I[0] = Vd
    I[-1] = 0
    for i in range(1,len(V)-1):
        I[i] = Cs[i-1]/dt * (V[i-1]-V[i]) - Cs[i]/dt  * (V[i]-V[i+1])
    return I

def find_current(V, R, Cs, dt):
    # calculates the current as a function of time
    # through resistors R and capacitors Cs
    V = np.array(V)
    R = np.array(R)
    Cs = np.array(Cs)
    
    Vdiff = -np.diff(V, axis = 1)
    IR_list = Vdiff / R

    # calculate derivative of voltage over time
    dVdiff_dt = np.diff(Vdiff, axis = 0)/dt
    # fill first column with zeros to avoid mismatching sizes
    dVdiff_dt = np.insert(dVdiff_dt, 0, 0, axis = 0)
    IC_list = (dVdiff_dt * Cs)
    return IR_list, IC_list

def modified_nodal_analysis(t0, dt, tmax, R_funcs, Cs, Vd):
    # performs modified nodal analysis on a circuit with
    # a driving voltage Vd(t) and n blocks of nonlinear 
    # resistive elements R_funcs in parallel with capacitors Cs.
    # 
    # R_funcs is a list of n functions R(V) for each resistive element
    # Cs is a list of n capacitances
    # Vd is a function of of time returning the voltage source value at t
    # 
    # returns list of voltages at each node in the circuit as a function
    # of time, and a list of current sources at each node as a funtion
    # of time

    V_list, t_list, R_list = [], [], []

    t0 = 0
    V = np.zeros(len(R_funcs) + 1)
    V[0] = Vd(t0)
    
    # find equivalent resistances of capacitor companion model
    RCeqs = dt/Cs
    for t in np.arange(t0, tmax, dt):
        # update nonlinear resistances according to current voltages
        Rs = update_R(R_funcs, V)
        # fill in the admittance matrix according to modified nodal analysis
        G = create_2D_device_G_matrix(Rs, RCeqs)
        # fill in current sources according to modified nodal analysis
        I = create_2D_device_current_sources(Cs, V, Vd(t), dt)
            
        # store values before updating
        V_list.append(V)
        t_list.append(t)
        R_list.append(Rs)

        # find resistance matrix from admittance matrix
        R = np.linalg.inv(G)
        # update voltage values at each node
        V = R @ I
        
    Rs = update_R(R_funcs, V)
    # fill in the admittance matrix according to modified nodal analysis
    G = create_2D_device_G_matrix(Rs, RCeqs)
    # fill in current sources according to modified nodal analysis
    I = create_2D_device_current_sources(Cs, V, Vd(t), dt)
    # store final values
    V_list.append(V)
    R_list.append(Rs)
    
    IR_list, IC_list = find_current(V_list, R_list, Cs, dt)
    
    return np.array(t_list), np.array(V_list[1:]), np.array(IR_list[1:]), np.array(IC_list[1:]), np.array(R_list[1:])

def inspect_voltages_at_nodes(x, Vs, d=None, layer_names = None, xlabel = 'Time (s)', xlog = False):
    for ind, v in enumerate(np.transpose(Vs)):
        plt.plot(x, v)
        plt.ylabel('Voltage (V)')
        plt.xlabel(xlabel)
        plt.gca().twinx().plot(x, v/d[ind]*1e-6)
        plt.ylabel('E-field (V/um)')
        if layer_names is not None:
            plt.title(layer_names[ind])
        if xlog:
            plt.xscale('log')
        plt.show()

def modified_nodal_analysis_amplitude_scan(t0, dt, tmax, R_funcs, Cs, f, Vapplied_list):
    Vmax_list, Vmin_list, Imax_list, Imin_list = [], [], [], []
    nsteps = tmax / dt

    for Vapplied in Vapplied_list:
#         Vd = lambda t : Vapplied * (0.5 + 0.5 * np.sin(2*np.pi*f*t + 0.2))
        Vd = lambda t : Vapplied * np.sin(2*np.pi*f*t)

        t, Vs, IRs, ICs, Rs= modified_nodal_analysis(t0, dt, tmax, R_funcs, Cs, Vd)

        Vdiff = np.diff(Vs)
        Is = IRs + ICs
        
        # we might have to select later indices of Vdiff and Is to avoid using the
        # transient solution
        Vmax_list.append(Vdiff[int(nsteps - 2/f/dt):].max(axis = 0))
        Vmin_list.append(Vdiff[int(nsteps - 2/f/dt):].min(axis = 0))
        Imax_list.append(Is[int(nsteps - 2/f/dt):].max(axis = 0))
        Imin_list.append(Is[int(nsteps - 2/f/dt):].min(axis = 0))

    return np.array(Vmax_list), np.array(Vmin_list), np.array(Imax_list), np.array(Imin_list)

def modified_nodal_analysis_frequency_scan(ncycles, npercycle, R_funcs, Cs, f_list, Vapplied):
    Vmax_list, Vmin_list, Imax_list, Imin_list = [], [], [], []
    nsteps = ncycles * npercycle

    for f in f_list:
        t0 = 0
        tmax = ncycles/f
        dt = 1/f/npercycle
        
#         Vd = lambda t : Vapplied * (0.5 + 0.5 * np.sin(2*np.pi*f*t + 0.2))
        Vd = lambda t : Vapplied * np.sin(2*np.pi*f*t)

        t, Vs, IRs, ICs, Rs= modified_nodal_analysis(t0, dt, tmax, R_funcs, Cs, Vd)

        Vdiff = np.diff(Vs)
        Is = IRs + ICs
        
        # we might have to select later indices of Vdiff and Is to avoid using the
        # transient solution
        Vmax_list.append(Vdiff[int(nsteps - 2/f/dt):].max(axis = 0))
        Vmin_list.append(Vdiff[int(nsteps - 2/f/dt):].min(axis = 0))
        Imax_list.append(Is[int(nsteps - 2/f/dt):].max(axis = 0))
        Imin_list.append(Is[int(nsteps - 2/f/dt):].min(axis = 0))

    return np.array(Vmax_list), np.array(Vmin_list), np.array(Imax_list), np.array(Imin_list)

def modified_nodal_analysis_amplitude_scan_dark_bright_diff(t0, dt, tmax, R_dark_funcs, R_bright_funcs, Cs, f, Vapplied_list):
    Vmax_list, Vmin_list, Imax_list, Imin_list = [], [], [], []
    nsteps = tmax / dt

    for Vapplied in Vapplied_list:
#         Vd = lambda t : Vapplied * (0.5 + 0.5 * np.sin(2*np.pi*f*t + 0.2))
        Vd = lambda t : Vapplied * np.sin(2*np.pi*f*t)

        t, V_darks, IR_darks, IC_darks, R_darks= modified_nodal_analysis(t0, dt, tmax, R_dark_funcs, Cs, Vd)
        Vdiff_dark = np.diff(V_darks)
        I_darks = IR_darks + IC_darks
        
        t, V_brights, IR_brights, IC_brights, R_brights= modified_nodal_analysis(t0, dt, tmax, R_bright_funcs, Cs, Vd)
        Vdiff_bright = np.diff(V_brights)
        I_brights = IR_brights + IC_brights
        
        # we might have to select later indices of Vdiff and Is to avoid using the
        # transient solution
        Vmax_list.append((Vdiff_dark-Vdiff_bright)[int(nsteps - 2/f/dt):].max(axis = 0))
        Vmin_list.append((Vdiff_dark-Vdiff_bright)[int(nsteps - 2/f/dt):].min(axis = 0))
        Imax_list.append((I_darks-I_brights)[int(nsteps - 2/f/dt):].max(axis = 0))
        Imin_list.append((I_darks-I_brights)[int(nsteps - 2/f/dt):].min(axis = 0))

    return np.array(Vmax_list), np.array(Vmin_list), np.array(Imax_list), np.array(Imin_list)

def modified_nodal_analysis_frequency_scan_dark_bright_diff(ncycles, npercycle, R_dark_funcs, R_bright_funcs, Cs, f_list, Vapplied):
    Vmax_list, Vmin_list, Imax_list, Imin_list = [], [], [], []
    nsteps = ncycles * npercycle

    for f in f_list:
        t0 = 0
        tmax = ncycles/f
        dt = 1/f/npercycle
        
#         Vd = lambda t : Vapplied * (0.5 + 0.5 * np.sin(2*np.pi*f*t + 0.2))
        Vd = lambda t : Vapplied * np.sin(2*np.pi*f*t)

        t, V_darks, IR_darks, IC_darks, R_darks= modified_nodal_analysis(t0, dt, tmax, R_dark_funcs, Cs, Vd)
        Vdiff_dark = np.diff(V_darks)
        I_darks = IR_darks + IC_darks
        
        t, V_brights, IR_brights, IC_brights, R_brights= modified_nodal_analysis(t0, dt, tmax, R_bright_funcs, Cs, Vd)
        Vdiff_bright = np.diff(V_brights)
        I_brights = IR_brights + IC_brights
        
        # we might have to select later indices of Vdiff and Is to avoid using the
        # transient solution
        Vmax_list.append((Vdiff_dark-Vdiff_bright)[int(nsteps - 2/f/dt):].max(axis = 0))
        Vmin_list.append((Vdiff_dark-Vdiff_bright)[int(nsteps - 2/f/dt):].min(axis = 0))
        Imax_list.append((I_darks-I_brights)[int(nsteps - 2/f/dt):].max(axis = 0))
        Imin_list.append((I_darks-I_brights)[int(nsteps - 2/f/dt):].min(axis = 0))
        
    return np.array(Vmax_list), np.array(Vmin_list), np.array(Imax_list), np.array(Imin_list)