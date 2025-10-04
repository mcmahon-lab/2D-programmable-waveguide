"""
Physics-Aware Training (PAT) utilities.

This module provides custom PyTorch autograd functions for physics-aware training,
enabling gradient propagation through physical systems or simulations with separate
forward and backward functions.
"""

import torch
def make_pat_func(f_forward, f_backward):
    """
    Construct a custom PyTorch autograd function for physics-aware training (PAT).

    Creates a differentiable function that uses separate forward and backward passes,
    enabling gradient-based optimization through physical experiments or mismatched simulations.
    The forward pass typically represents a physical system, while the backward pass uses
    a model (possibly simplified or idealized) for gradient estimation.

    Inputs:
    -------
    f_forward : callable
        Function applied during forward pass, signature: f_forward(*args) → output.
        Typically represents a physical experiment or high-fidelity simulation.
        For demo code: f_forward(x, theta_1, theta_2, ...) where x is input and theta_i are parameters.
    f_backward : callable
        Function used during backward pass for gradient estimation, signature: f_backward(*args) → output.
        Must have same signature as f_forward. Typically a differentiable model/simulation.

    Returns:
    --------
    f_pat : callable
        Custom autograd function with PAT behavior, signature: f_pat(*args) → output.
        Can be used like a standard PyTorch function in training loops.

    Note:
    -----
    The forward pass stores all inputs for the backward pass. During backpropagation,
    vector-Jacobian product (vjp) is computed using f_backward, not f_forward.
    """
    class func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args): 
            ctx.save_for_backward(*args)
            return f_forward(*args)
        def backward(ctx, grad_output):
            args = ctx.saved_tensors
            torch.set_grad_enabled(True)
            y = torch.autograd.functional.vjp(f_backward, args, v=grad_output)
            torch.set_grad_enabled(False)
            return y[1]
    return func.apply