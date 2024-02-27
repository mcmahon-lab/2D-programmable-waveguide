from pylablib.devices import Thorlabs
from tdwg.lib.DMD_patterns import generate_all_on
from tdwg.lib.beams_utils import get_gaussian
import numpy as np
import matplotlib.pyplot as plt
import time
import astropy.units as u
# Thorlabs.list_kinesis_devices()

class TDWG_piezo_controller(Thorlabs.KinesisMotor):
    def __init__(self, 
                 serial_number = 27258848,
                 step_size = 15,
                 *args, 
                 **kwargs,
                ):   
        super().__init__(conn = serial_number, *args, **kwargs)
        
        # choose -1 because the power optimization procedure ends on -1. 
        # This will not always be right, but most of the time if we don't manually move the motor
        self.last_direction = -1 
        self.step_size = step_size
        self.setup_jog(step_size = self.step_size,
                       mode = 'step')
        self.is_compensated = False
        
    def jog(self, direction, timeout_s = 0.2):
        """Takes a single jogging step in the direction specified (direction = +1 or -1)"""
        self.last_direction = direction
        self.is_compensated = False
        if direction == -1: direction  = "-"
        if direction == 1: direction  = "+"
        # kind = "builtin" guarantees that the motor does not continuously jog in one direction
        super().jog(direction, kind = "builtin")
        time.sleep(timeout_s) # I found that anything shorter results in unstable behavior
        
    def compensate_backlash(self, direction, n_steps = 12):
        """Compensate for hysteresis of motor by taking n_steps opposite of the 
        last registered direction. n_steps=12 works well for default parameters"""
        last_direction = self.last_direction
        if not direction == self.last_direction:
            if not self.is_compensated:
                for i in range(n_steps):
                    self.jog(-last_direction)
                self.is_compensated = True
        
    def jog_n_steps(self, n, timeout_s = 0.2):
        """Take n jogging steps. The sign of n determines in which direction to jog."""
        direction = np.sign(n)
        n_steps = np.abs(n)
        for i in range(n_steps):
            self.jog(direction)
            
    def jog_n_steps_compensated(self, n, timeout_s = 0.2):
        """Take n jogging steps. The sign of n determines in which direction to jog.
        If the previous jogging direction was opposite of the requested direction,
        the hysteresis compensation is applied first."""
        direction = np.sign(n)
        self.compensate_backlash(direction)
        self.jog_n_steps(n, timeout_s)
        
    ### From here on, the functions are very 2D-waveguide specific. Can be deleted for other future projects.
        
    def jog_n_steps_compensated_monitored(self, linecam, n, timeout_s = 0.2):
        """Take n jogging steps. The sign of n determines in which direction to jog.
        If the previous jogging direction was opposite of the requested direction,
        the hysteresis compensation is applied first. Monitors camera output and 
        overlap with a supplied cladding and core mode."""
        output_list = []
        
        direction = np.sign(n)
        n_steps = np.abs(n)
        self.compensate_backlash(direction)
        for i in range(n_steps):
            self.jog(direction, timeout_s=timeout_s)
            output = linecam.get_output()
            output_list.append(output)
        
        return np.array(output_list)
    
    def set_calibration_beam(self, beamshaper, xcenter = 0*u.um, w0 = 20*u.um, fx = 0/(1*u.mm)):
        """Creates a beam on the beamshaper for which cladding and core modes are maximally separated (large fx)"""
        Ein_x_bs = get_gaussian(beamshaper.x_bs, xcenter, w0, fx)
        beamshaper.apply_Ein(Ein_x_bs, 0.05)
        
    def set_calibration_DMD_img(self, dmd, img = generate_all_on()):
        """Sets a standard DMD image ()"""
        dmd.apply_image_hold(img)
        time.sleep(0.3)
        
    def maximize_core_overlap(self, linecam, sweep_steps = 40, diagnostics = True):
        """Sweeps the z-direction with a reasonable amount of sweep steps and then moves the motor to the position
        with the maximal power"""
        # move to negative end
        self.jog_n_steps_compensated(-int(sweep_steps/2))

        # sweep power in positive direction
        output_list = self.jog_n_steps_compensated_monitored(linecam, sweep_steps)
        output_power = output_list.sum(-1)
        self.output_power = output_power

        # walk backwards until overall power decreases. 
        nsteps = sweep_steps - (output_list - 540).sum(-1).argmax() -3
        output_list_test = self.jog_n_steps_compensated_monitored(linecam, -nsteps)
        output_power_old = output_list_test.sum(-1)[-1]
        while True:
            nsteps += 1
            output_list_new = self.jog_n_steps_compensated_monitored(linecam, -1)
            output_list_test = np.concatenate((output_list_test, output_list_new))
            output_power_new = output_list_new.sum()
            if output_power_new < output_power_old:
                break
            else:
                output_power_old = output_power_new
        output_power_test = output_list_test.sum(-1)
        #########

        if diagnostics:
            power_ratio = (output_power_test[-1] - output_power.min()) / (output_power.max() - output_power.min())
            print(f'Reached {100*power_ratio:.1f}% of maximal power')

            plt.plot(output_power, '.-', c = 'tab:orange', alpha = 0.5)
            plt.plot(np.arange(sweep_steps, sweep_steps-nsteps, -1), output_power_test, '.-', c = 'tab:blue')
    
    def monitor_output_power(self, beamshaper, linecam, dmd):
        self.set_calibration_beam(beamshaper)
        self.set_calibration_DMD_img(dmd)
        output = linecam.get_output()
        output_power = output.sum()
        power_ratio = (output_power - self.output_power.min()) / (self.output_power.max() - self.output_power.min())
        return power_ratio