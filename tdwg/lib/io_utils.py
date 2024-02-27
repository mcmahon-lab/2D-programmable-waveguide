"""
This module is very very OP. Honestly, a lot of the functions are unsafe, so becareful when using them! The reason is that the functions here all have the ability to edit the global varibles. 

To use any of the functions, global_dict=globals() must be passed. 
"""

from importlib import reload
import types
import pickle
import inspect
import matplotlib
import os
import ctypes
import numpy
import astropy

def crazy_reload(package_string, global_dict):
    """
    For this function to work, you MUST run it as follows:
    crazy_reload(package_string, globals())
    """
    exec(rf"import {package_string}")
    exec(rf"reload({package_string})")
    exec(rf"from {package_string} import *")
    local_vars = locals()
    del local_vars["global_dict"]
    global_dict.update(local_vars)
    
def get_user_vars(global_dict, ignore_vars=None):
    """
    Get the variables that have been defined in a session!
    It is pretty handy for saving all of the variables in one go!

    This is a much safer version than the one that I had before...
    """
    data_dict = dict()

    var_ignore_set = ['In', 'Out', 'get_ipython', 'exit', 
                  'quit', "package_string", "RTLD_LOCAL", "RTLD_GLOBAL", "DEFAULT_MODE"]
    if ignore_vars is not None:
        var_ignore_set = var_ignore_set + ignore_vars

    for key, val in global_dict.items():
        cond_list = []
        cond_list.append(not key.startswith('_'))
        cond_list.append(key not in var_ignore_set)

        if all(cond_list):
            if isinstance(val, (int, float, str, bool, complex, list, dict, numpy.ndarray, astropy.units.quantity.Quantity)):
                if is_picklable(val):
                    #putting it here so that the is_picklable function is called as few times as possible
                    data_dict[key] = val
    return data_dict

def is_picklable(obj):
    try:
        # Attempt to pickle the object
        pickle.dumps(obj)
        return True
    except:
        return False

def pickle_all_data(file, global_dict, ignore_vars=None, print_flag=True):
    """
    This function will take all of the data that is present in your current active session, and save it in pickle file. 
    as usual you must use the function like this
    pickle_all_data("haha.pkl", globals()), otherwise, it won't work!
    """
    if os.path.exists(file):
        raise ValueError(f"{file} already exists! Please delete it first!")

    data_dict = get_user_vars(global_dict, ignore_vars)

    if print_flag:
        print("The following variables are pickled!")
        for key in data_dict.keys():
            print(f"{key}, ", end="")

    with open(file, "wb") as f:
        pickle.dump(data_dict, f)
        
def unpickle_all_data(file, global_dict, print_flag=True):
    """
    This function is meant to work together with the function pickle_all_data! 
    It will load up the data_dict in that dictionary, and then populate all of the global namespace with the variables in that dictionary. 
    
    This function is VERY VERY OP - becareful when using it! 
    unpickle_all_data("haha.pkl", globals()) is as usual the recommended usage for the code
    """
    with open(file, "rb") as f:
        data_dict = pickle.load(f)
    
    if print_flag:
        print("The following variables are being loaded into the global namespace:")
        for key in data_dict.keys():
            print(f"{key}, ", end="")
    
    global_dict.update(data_dict)

def unpickle_some_data(file, vars, global_dict, print_flag=True):
    with open(file, "rb") as f:
        data_dict = pickle.load(f)

    #throw an error if there is a variable in vars that is not in the data_dict
    for var in vars:
        if var not in data_dict.keys():
            raise ValueError(f"{var} is not in the data_dict!")

    data_dict = {key: value for key, value in data_dict.items() if key in vars}
    global_dict.update(data_dict)

def print_variables_data_dict(file):
    with open(file, "rb") as f:
        data_dict = pickle.load(f)

    print("Data dict has the following variables:")
    for key in data_dict.keys():
        print(f"{key}, ", end="")
        
def pickle_data_dict(file, data_dict, override_flag=False):
    # The code does a check if there is a file that already exist, this is the safest way to run things...
    if not override_flag:
        if os.path.isfile(file):
            raise Exception("The file already exist!")
        
    with open(file, "wb") as f:
        pickle.dump(data_dict, f)