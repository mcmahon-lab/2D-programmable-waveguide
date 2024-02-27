__all__ = ['sweep_zip', 'sweep_product']

import numpy as np
import itertools
from collections.abc import Iterable
import ipdb

from progressbar import ProgressBar, SimpleProgress, Bar, Timer, Percentage

######## Most import functions, provided by import ##########

def sweep_zip(func, *args_list, verbose=True):
    """
    Applies a function to the arguments, which are zipped over.
    """
    result_list = zip_single_out(func, *args_list, verbose=verbose)
    
    multi_output_flag = isinstance(result_list[0], tuple)
    if multi_output_flag:
        #correspond to the multi-output functions
        result_dim = len(result_list[0])
        output = []
        for dim_ind in range(result_dim):
            output_list = []
            for result in result_list:
                output_list.append(result[dim_ind])
            output.append(output_list)
        output = tuple(output)
    else:
        #single output functions!
        output = result_list
    return output

def sweep_product(func, *args, verbose=True):
    """
    Applies a function to the arguments, with a list of Cartesian product inputs. 
    
    Example usage:
    xlist = [1, 3, 5]
    ylist = [3, 10]
    
    def main_mimo(x, y):
        time.sleep(0.1)
        out1 = np.ones(10)*y
        return out1, x*y
    
    out1, out2 = sweep_product(main_mimo, xlist, ylist)

    Note that the outer loop happens over the first argument (xlist in the previous example).
    So if you want to test for repeatability, define the function so the "seed" variable is the first argument.
    See craft notes on 2022-06-25 for how I tested this.
    """
    args_list = list(itertools.product(*args))
    dims = [len(arg) for arg in args]

    result_list = []
    if verbose:
        widgets = [Percentage(), ' ', SimpleProgress(), ' ', Bar(), ' ', Timer()]
        pbar = ProgressBar(widgets=widgets, max_value=len(args_list))
    for i, arg in enumerate(args_list):
        result = func(*arg)
        result_list.append(result)
        if verbose:
            pbar.update(i+1)
    if verbose:
        pbar.finish()

    multi_output_flag = isinstance(result_list[0], tuple)
    if multi_output_flag:
        #multi output
        Noutputs = len(result_list[0])
        result_array_list = [np.empty(dims, dtype=object) for o_ind in range(Noutputs)]
        for i, result in enumerate(result_list):
            for o_ind in range(Noutputs):
                result_array_list[o_ind].flat[i] = result[o_ind]

    else:
        #single output
        result_array_list = np.empty(dims, dtype=object)
        for i, result in enumerate(result_list):
            result_array_list.flat[i] = result

    return result_array_list


###### These other ones are internal functions (not exported) ###############33

def zip_single_out(func, *args_list, verbose=True):
    """
    Use this if the output of the function that has a single output
    """
    result_list = []
    if verbose:
        widgets = [Percentage(), ' ', SimpleProgress(), ' ', Bar(), ' ', Timer()]
        pbar = ProgressBar(widgets=widgets, max_value=len(args_list[0]))
    for i, arg in enumerate(zip(*args_list)):
        result = func(*arg)
        result_list.append(result)
        if verbose:
            pbar.update(i+1)
    if verbose:
        pbar.finish()
    return result_list

