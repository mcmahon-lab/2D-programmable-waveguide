## Comment by Martin:
## This file is originally from pymeasure for the Tektronix AFG1022
## We are using a Tektronix AFG3102C in our setup, but all commands
## used in this file are the same and verified to work (2023-02-11).
## Some arguments like the compliance voltage have a hardware implementation
## on the Tektronix AFG3102C which we SHOULD use in the future but this file
## does not implement yet.
#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2019 PyMeasure Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

# from pymeasure.instruments import Instrument

import pyvisa as visa
import numpy as np
import time
import json
from io import StringIO
from tdwg.lib.misc_utils import arange_inclusive

class AFG():
    def __init__(self, 
                 compliance_voltage_channel1 = 0.02, 
                 compliance_voltage_channel2 = 0.02, 
                 resourceName = 'USB0::0x0699::0x034C::C010755::INSTR', 
                 **kwargs):
        rm = visa.ResourceManager()
        self.instrument = rm.open_resource(resourceName)
        print('Connected to ', self.instrument.query("*IDN?"))
        self.channel1 = AFG.Channel(self, channel = 1, compliance_voltage = compliance_voltage_channel1)
        self.channel2 = AFG.Channel(self, channel = 2, compliance_voltage = compliance_voltage_channel2)
        
    class Channel():
        def __init__(self, parent, channel, compliance_voltage):
            self.channel = channel
            self.parent = parent.instrument
            if compliance_voltage == 0.02:
                self.set_compliance_voltage(compliance_voltage, skip = True)
            else: 
                self.set_compliance_voltage(compliance_voltage)
    
        def get_compliance_voltage(self):
            return self._compliance_voltage
            
        def set_compliance_voltage(self, value, skip = False):
            print(f"ATTENTION: Compliance voltage for channel {self.channel} will be set to {value} V.")
            if skip:
                self._compliance_voltage = value 
            else:
                user_input = input(f"Confirm by typing {int(200*value)}, abort with any other key: ")
                if int(user_input) == int(200*value):
                    self._compliance_voltage = value 
                    print(f'Compliance voltage set to {value} V.')
                else:
                    print('Compliance voltage change aborted.')
                
        @property
        def function_shape(self):
            return self.parent.query(f"SOUR{self.channel}:FUNC:SHAP?")
        
        @function_shape.setter
        def function_shape(self, value):
            values = ['SIN','SQU','PUL','RAMP','PRN','DC','SINC','GAUS']
            if value in values:
                self.parent.write(f"SOUR{self.channel}:FUNC:SHAP %s" % value)
            """
            Sets or queries the amplitude modulation of the output function for channel.

            - Values: :code:`SIN`, :code:`SQU`, :code:`PUL`, :code:`RAMP`, :code:`PRN`, :code:`DC`, :code:`SINC`, :code:`GAUS`,
            """            

        @property
        def ramp_symmetry(self):
            return float(self.parent.query(f"SOUR{self.channel}:FUNC:RAMP:SYMM?"))
        
        @ramp_symmetry.setter
        def ramp_symmetry(self, value):
            self.parent.write(f"SOUR{self.channel}:FUNC:RAMP:SYMM %e" % value)
            """ Sets or queries the symmetry of ramp waveform. The setting range is 0 to 100%."""

        @property
        def frequency(self):
            return float(self.parent.query(f"SOUR{self.channel}:FREQ:CW?"))
        
        @frequency.setter
        def frequency(self, value):
            self.parent.write(f"SOUR{self.channel}:FREQ:CW %e" % value)
            
        @property
        def phase_rad(self):
            return float(self.parent.query(f"SOUR{self.channel}:PHASE?"))
        
        @phase_rad.setter
        def phase_rad(self, value):
            self.parent.write(f"SOUR{self.channel}:PHASE %e" % value)
            
        @property
        def phase_deg(self):
            return np.rad2deg(float(self.parent.query(f"SOUR{self.channel}:PHASE?")))
        
        @phase_deg.setter
        def phase_deg(self, value):
            self.parent.write(f"SOUR{self.channel}:PHASE %e" % np.deg2rad(value))

        @property
        def offset(self):
            return float(self.parent.query(f"SOUR{self.channel}:VOLT:LEV:IMM:OFFS?"))
        
        @offset.setter
        def offset(self, value):
            if np.abs(value) <= np.abs(self._compliance_voltage) - np.abs(self.voltage):
                self.parent.write(f"SOUR{self.channel}:VOLT:LEV:IMM:OFFS %e" % value)
            else: 
                print(f"Requested offset of {value} V is exceeding complicance voltage |{self._compliance_voltage}| V minus amplitude voltage |{self.voltage}V|. Voltage remains unchanged.")
            
        @property
        def voltage(self):
            return float(self.parent.query(f"SOUR{self.channel}:VOLT:LEV:IMM:AMPL?"))
        
        @voltage.setter
        def voltage(self, value):
            if np.abs(value) <= np.abs(self._compliance_voltage) - np.abs(self.offset):
                self.parent.write(f"SOUR{self.channel}:VOLT:LEV:IMM:AMPL %e" % value)
            else: 
                print(f"Requested voltage of {value} V is exceeding complicance voltage |{self._compliance_voltage} V| minus offset voltage |{self.offset} V|. Voltage remains unchanged.")

        def turn_on(self):
            self.parent.write(f"OUTP{self.channel}:STAT ON")

        def turn_off(self):
            self.parent.write(f"OUTP{self.channel}:STAT OFF")
            
        def is_on(self):
            return bool(float(self.parent.query(f"OUTP{self.channel}:STAT?")))
        
        def to_json(self,file_name):
            params = {}
            params['voltage'] = self.voltage
            params['frequency'] = self.frequency
            params['function_shape'] = self.function_shape
            with open(file_name,'w') as f: json.dump(params,f)
        

class TDWG_AFG(AFG):
    def __init__(self, 
                 compliance_voltage_channel1 = 0.02, 
                 compliance_voltage_channel2 = 2.6, 
                 resourceName = 'USB0::0x0699::0x034C::C010755::INSTR', 
                 **kwargs):
        super().__init__(compliance_voltage_channel1, compliance_voltage_channel2, resourceName, **kwargs)
        
    def set_compliance_voltage(self, voltage):
        """
        High voltage is on channel 1
        LED is on channel 2, the compliance voltage don't change so it is hardcoded in the source code.
        """
        self.channel1.set_compliance_voltage(voltage)
        self.channel2.set_compliance_voltage(2.6, skip=True)

    def turn_on_high_voltage(self, voltage):
        """Turns on high voltage"""
        self.channel1.turn_on()
        time.sleep(1)
        sign = np.sign(voltage - self.channel1.voltage, )
        if voltage == self.channel1.voltage: return
        for voltage in arange_inclusive(self.channel1.voltage, voltage, sign*0.1):
            self.channel1.voltage = voltage
            time.sleep(0.2)

    def turn_off_high_voltage(self):
        """Turns off high voltage"""
        for voltage in arange_inclusive(self.channel1.voltage, 0.1, -0.1):
            self.channel1.voltage = voltage
            time.sleep(0.2)
        time.sleep(1)
        self.channel1.turn_off()
        time.sleep(0.5)

    def turn_on_led(self):
        """Turns on LED"""
        self.channel2.turn_on()
        time.sleep(1)
        self.channel2.offset = 1.

    def turn_off_led(self):
        """Turns off LED"""
        self.channel2.turn_on()
        time.sleep(1)
        self.channel2.offset = 0.

    def get_afg_voltage(self):
        """
        Returns the current voltage that is applied by the self.towards the high voltage.
        Multiply by 200X to get the applied voltage or run the get_high_voltage_trace to see the measured quantity.
        """
        return self.channel1.voltage