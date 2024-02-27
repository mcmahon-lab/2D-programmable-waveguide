import numpy as np
from tdwg.lib.ALP4b import *

from tdwg.lib.multiuser_client import Client
client = Client()

from copy import deepcopy #for image rewriting stuff

class DMDCore():
    def __init__(self):
        self.instrument = ALP4(version = '4.2V', libDir = 'C:/Program Files/ALP-4.2/ALP-4.2 high-speed API')
        self.instrument.Initialize()
        self.image_on_display = None
        
    def apply_image_hold(self, img, time_in_s=0.005, loop = True, bitDepth = 8):
        if self.image_on_display is not None:
            self.stop_image_hold()
        imgSeq  = np.concatenate([img.ravel()])
        self.instrument.SeqAlloc(nbImg = 1, bitDepth = bitDepth)
        self.instrument.SeqPut(imgData = imgSeq)
        self.instrument.SetTiming(pictureTime = int(time_in_s*1e6))

        self.instrument.Run(loop = loop) #display the sequence in a loop
        self.image_on_display = img

    def stop_image_hold(self):
        self.instrument.Halt() 
        self.instrument.FreeSeq()


class DMD():
    def __init__(self, mode = "multiuser", calibration_dict = dict(dmd_safety_margin = 520)):
        """
        multiuser: This assumes that a server kernel must be running!
        local: This means that no other instance of beamshaper is allowed at a given time (old code)
        """
        self.mode = mode
        if mode == "local":
            self.dmd_core = DMDCore()
            
        self.load_calibration(calibration_dict)
            
    def load_calibration(self, calibration_dict):
        self.safety_margin = calibration_dict['dmd_safety_margin'] 
    
    def apply_image_hold(self, img, time_in_s=0.005, loop = True, bitDepth = 8):
        img = deepcopy(img)
        img[:self.safety_margin] = 0
        args = [img, time_in_s, loop, bitDepth]
        if self.mode == "local":
            self.dmd_core.apply_image_hold(*args)
        else:
            client.run_command("dmd_core.apply_image_hold", args)
        
    def stop_image_hold(self):
        if self.mode == "local":
            self.dmd_core.stop_image_hold()
        else:
            client.run_command("dmd_core.stop_image_hold")
      
        
        
        