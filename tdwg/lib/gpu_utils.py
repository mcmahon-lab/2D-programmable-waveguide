"""
This file will contain function that allows for easier management of GPU memory.

To clear all memory, copy paste this code in!
Amazingly, it won't work as well if you wrap this in a function...

for i in range(2):
    clear_gpu_references(device, globals())
    empty_cache()
"""
import gc
from collections import OrderedDict
import torch

def kill_kernel():
    """
    This function will kill the current kernel.
    """
    import os
    os._exit(00)

def empty_cache():
    """
    Mostly here to help with remembering the command.
    """
    torch.cuda.empty_cache()

def remove_all_references(obj):
    # Get the referrers of the object
    referrers = gc.get_referrers(obj)
    
    # For each referrer
    for ref in referrers:
        # If the referrer is a list or a set, try to remove the object
        if isinstance(ref, (list, set)):
            ref[:] = [x for x in ref if x is not obj]
        # If the referrer is a dictionary, try to remove the object from the values
        elif isinstance(ref, dict):
            for key, value in list(ref.items()):
                if value is obj:
                    del ref[key]

def find_tensors(device, global_dict):
    del_list = []
    max_depth = 5
    seen_objects = set()

    # If it's a tensor, delete it
    def delete_tensors(obj, depth=0):
        if depth > max_depth:
            return
        if id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))

        # If it's a tensor on the right device, add it to del_list
        if torch.is_tensor(obj) and obj.device == torch.device(device):
            del_list.append(obj)

        # First check if it is a dictionary or ordered dictinary
        if isinstance(obj, dict) or isinstance(obj, OrderedDict):
            for val in obj.values():
                delete_tensors(val, depth+1)

        if hasattr(obj, '__dict__'):
            if len(obj.__dict__) > 0:
                for val in obj.__dict__.values():
                    delete_tensors(val, depth+1)

        #explicitly, I decided against "dealing" with iterables. This is because it can lead to scary scenarios if one is not careful. (Like looping over a large numpy array...). Will add it later if necessary!

    for name, val in list(global_dict.items()): 
        delete_tensors(val)
    return del_list

def clear_gpu_references(device, global_dict):
    del_list = find_tensors(device, global_dict)
    for obj in del_list:
        remove_all_references(obj)