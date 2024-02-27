#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2021 PyMeasure Developers
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

import pyvisa as visa
import numpy as np
from struct import unpack
import matplotlib.pyplot as plt

class Oscilloscope():
    """ Represents the Tektronix TBS 1104 
    and provides a high-level for interacting with the instrument
    """

    class Measurement(object):

        SOURCE_VALUES = ['CH1', 'CH2', 'CH3', 'CH4', 'MATH']

        TYPE_VALUES = [
            'FREQ', 'MEAN', 'PERI', 'PHA', 'PK2', 'CRM',
            'MINI', 'MAXI', 'RIS', 'FALL', 'PWI', 'NWI'
        ]

        UNIT_VALUES = ['V', 's', 'Hz']

        def __init__(self, parent, preamble="MEASU:IMM:"):
            self.parent = parent.instrument
            self.preamble = preamble

        @property
        def value(self):
            return float(self.parent.query("%sVAL?" % self.preamble))

        @property
        def source(self):
            return self.parent.query("%sSOU?" % self.preamble).strip()

        @source.setter
        def source(self, value):
            if value in Oscilloscope.Measurement.SOURCE_VALUES:
                self.parent.write("%sSOU %s" % (self.preamble, value))
            else:
                raise ValueError("Invalid source ('%s') provided to %s" % (
                                 self.parent, value))

        @property
        def type(self):
            return self.parent.query("%sTYP?" % self.preamble).strip()

        @type.setter
        def type(self, value):
            if value in Oscilloscope.Measurement.TYPE_VALUES:
                self.parent.write("%sTYP %s" % (self.preamble, value))
            else:
                raise ValueError("Invalid type ('%s') provided to %s" % (
                                 self.parent, value))

        @property
        def unit(self):
            return self.parent.query("%sUNI?" % self.preamble).strip()

        @unit.setter
        def unit(self, value):
            if value in Oscilloscope.Measurement.UNIT_VALUES:
                self.parent.write("%sUNI %s" % (self.preamble, value))
            else:
                raise ValueError("Invalid unit ('%s') provided to %s" % (
                                 self.parent, value))
                
    class Channel():
        def __init__(self, parent, channel):
            self.channel = channel
            self.parent = parent.instrument
            
        @property
        def y_scale(self):
            # returns voltage per division in volts 
            return float(self.parent.query(f'CH{self.channel}:SCALE?'))

        @y_scale.setter
        def y_scale(self, vdiv):
            return self.parent.write(f'CH{self.channel}:SCALE {vdiv}')

        @property
        def y_offset(self):
            # returns offset in volts 
            return float(self.parent.query(f'CH{self.channel}:POSITION?'))

        @y_offset.setter
        def y_offset(self, voff):
            return self.parent.write(f'CH{self.channel}:POSITION {voff}')

    def __init__(self, resourceName = 'USB0::0x0699::0x03B4::C020764::INSTR', **kwargs):
        rm = visa.ResourceManager()
        self.instrument = rm.open_resource(resourceName)
        print('Connected to ', self.instrument.query("*IDN?"))
        self.measurement = Oscilloscope.Measurement(self)
        self.channel1 = Oscilloscope.Channel(self, channel = 1)
        self.channel2 = Oscilloscope.Channel(self, channel = 2)
        self.channel3 = Oscilloscope.Channel(self, channel = 3)
        self.channel4 = Oscilloscope.Channel(self, channel = 4)
        
    def process_raw_data(self, data, yoff, ymult, yzero, xincr):
        headerlen = 2 + int(data[1])
        header = data[:headerlen]
        ADC_wave = data[headerlen:-1]
        ADC_wave = np.array(unpack('%sb' % len(ADC_wave),ADC_wave))
        volts = (ADC_wave - yoff) * ymult  + yzero
        time = np.arange(0, xincr * len(volts), xincr)
        return time, volts

    def get_trace(self, channel = 1, raw = False):
        # retrieve data from specified channel
        self.instrument.write(f'DATA:SOUR CH{channel}')
        self.instrument.write('DATA:WIDTH 1')
        self.instrument.write('DATA:ENC RIB')
        ymult = float(self.instrument.query('WFMPRE:YMULT?'))
        yzero = float(self.instrument.query('WFMPRE:YZERO?'))
        yoff = float(self.instrument.query('WFMPRE:YOFF?'))
        xincr = float(self.instrument.query('WFMPRE:XINCR?'))

        self.instrument.write('CURVE?')
        data = self.instrument.read_raw()

        time, volts = self.process_raw_data(data, yoff, ymult, yzero, xincr)
        if raw == True:
            return time, volts, data
        else:
            return time, volts
    
    @property
    def hor_scale(self):
        # returns time per division in seconds
        return float(self.instrument.query(f'HOR:SCALE?'))
    
    @hor_scale.setter
    def hor_scale(self, tdiv):
        # tdiv: time per division in seconds
        return self.instrument.write(f'HOR:SCALE {tdiv}')
    
    def print_screen(self):
        t, v1 = self.get_trace(1)
        t, v2 = self.get_trace(2)
        t, v3 = self.get_trace(3)
        t, v4 = self.get_trace(4)

        y1_scl = self.channel1.y_scale
        y1_off = self.channel1.y_offset
        y2_scl = self.channel2.y_scale
        y2_off = self.channel2.y_offset
        y3_scl = self.channel3.y_scale
        y3_off = self.channel3.y_offset
        y4_scl = self.channel4.y_scale
        y4_off = self.channel4.y_offset

        x_scl = self.hor_scale

        plt.style.use("dark_background")
        plt.plot(t[:len(v1)]/x_scl, v1/y1_scl + y1_off, '.', color = 'yellow', ms = 2, label = 'CH1')
        plt.plot(t[:len(v2)]/x_scl, v2/y2_scl + y2_off, '.', color = 'cyan', ms = 2, label = 'CH2')
        plt.plot(t[:len(v3)]/x_scl, v3/y3_scl + y3_off, '.', color = 'fuchsia', ms = 2, label = 'CH3')
        plt.plot(t[:len(v4)]/x_scl, v4/y4_scl + y4_off, '.', color = 'lime', ms = 2, label = 'CH4')

        plt.ylim(-4, 4)
        plt.xlim(0,10)
        plt.grid()
        plt.legend()
        plt.show()
        plt.style.use("default")
    
    
class TDWG_Oscilloscope(Oscilloscope):
    def __init__(self, resourceName = 'USB0::0x0699::0x03B4::C020764::INSTR', **kwargs):
        super().__init__(resourceName, **kwargs)
        
    def get_high_voltage_trace(self):
        """
        Returns the current high voltage
        Because the scope involves a 200X multiplication, this returns the actual applied high voltage

        outputs are t, voltage
        """
        return self.get_trace(2)

    def get_current_trace(self):
        """Returns the current high voltage"""
        return self.get_trace(3)

