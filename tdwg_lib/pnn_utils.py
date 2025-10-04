"""
Physics-informed Neural Network (PNN) utilities.

This module provides tools for training neural networks with physical constraints:
- Parameter class with enforced bounds
- Lagrangian penalty functions for constraint enforcement
- Loss functions and convergence detection
- Rescaled soft-ReLU/softplus function implementation

Uses PyTorch for neural network operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def L2(p, q):
    """
    Compute mean L2 distance between two tensors.

    Calculates the average squared difference between p and q, summing over the last
    dimension and averaging over all other dimensions.

    Inputs:
    -------
    p : torch.Tensor
        First tensor (arbitrary shape).
    q : torch.Tensor
        Second tensor (must have same shape as p).

    Returns:
    --------
    loss : torch.Tensor (scalar)
        Mean L2 distance: mean(sum(|p-q|²)) / 2.
    """
    x = torch.abs(p-q)**2
    return x.sum(dim = -1).mean()/2

def relu_approx(x, factor=20.0):
    """
    Compute smooth approximation to ReLU using rescaled softplus function.

    Implements: softplus(x*factor)/factor ≈ max(0, x) for large factor.
    Provides a differentiable alternative to ReLU for constraint enforcement.
    See: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html

    Inputs:
    -------
    x : torch.Tensor
        Input tensor (arbitrary shape).
    factor : float, default 20.0
        Sharpness parameter. Larger values → closer to ReLU, smaller values → smoother.

    Returns:
    --------
    output : torch.Tensor
        Smoothed ReLU approximation with same shape as x.
    """
    return F.softplus(x*factor)/factor

def clamp_lag(x, low=0.0, high=1.0, factor=20.):
    """
    Compute smooth penalty for values outside interval [low, high].

    Approximately returns the mean distance of elements of x from the interval [low, high].
    Values inside the interval contribute ~0 loss; values outside contribute proportional to
    their distance from the boundary. Higher factor → more accurate approximation.

    Inputs:
    -------
    x : torch.Tensor
        Input tensor for which penalty is calculated.
    low : float, default 0.0
        Lower boundary. Values below low by distance δ incur penalty ≈ δ.
    high : float, default 1.0
        Upper boundary. Values above high by distance δ incur penalty ≈ δ.
    factor : float, default 20.0
        Sharpness parameter. Larger values → sharper transition at boundaries.

    Returns:
    --------
    loss : torch.Tensor (scalar)
        Mean penalty for constraint violations.
    """
    return torch.mean(relu_approx(-(x-low), factor) + relu_approx(x-high, factor)) 


def lagrangian(model, lag_amp = 1., factor=20.0):
    """
    Compute Lagrangian penalty for model parameters violating their bounds.

    Iterates through all model parameters of type Parameter (which have .limits attribute),
    normalizes them to [0, 1] based on their limits, and adds penalty for values outside
    this range. Used during PNN training to enforce physical constraints.

    Inputs:
    -------
    model : torch.nn.Module
        Neural network model containing Parameter objects with .limits attributes.
    lag_amp : float, default 1.0
        Penalty multiplier. For each unit distance δ a parameter exceeds its bounds,
        approximately δ*lag_amp is added to the loss.
    factor : float, default 20.0
        Sharpness parameter for constraint enforcement (passed to clamp_lag).

    Returns:
    --------
    loss : torch.Tensor (scalar)
        Total Lagrangian penalty summed over all bounded parameters.
    """
    loss = 0.
    for parameter in model.parameters():
        if type(parameter) == Parameter:
            # normalize between 0 and 1
            p_norm = parameter - parameter.limits[0]
            p_norm = p_norm / (parameter.limits[1] - parameter.limits[0])
            loss += lag_amp*(clamp_lag(p_norm, 0, 1, factor))
    return loss

class Parameter(nn.Parameter):
    """
    Extended nn.Parameter that stores upper and lower bounds for constraint enforcement.

    Subclass of torch.nn.Parameter that adds a .limits attribute to specify valid
    parameter ranges. Used with lagrangian() function during PNN training to enforce
    physical constraints (e.g., refractive index bounds).
    """
    def __new__(cls, data=None, requires_grad=True, limits=None):
        """
        Create a bounded parameter.

        Inputs:
        -------
        data : torch.Tensor
            Initial parameter values.
        requires_grad : bool, default True
            Whether gradients should be computed for this parameter.
        limits : list or tuple of two floats, optional
            [lower_bound, upper_bound] specifying valid parameter range.
            Used by lagrangian() to penalize out-of-bounds values.

        Returns:
        --------
        param : Parameter
            Parameter object with .limits attribute.
        """
        param = nn.Parameter.__new__(cls, data = data, requires_grad=requires_grad)
        param.limits = limits
        return param

    def __repr__(self):
        return 'Parameter containing:\n' + super(nn.Parameter, self).__repr__() + '\tLimits: ' + str(self.limits)
        
def has_converged(loss_list, window=5, std_threshold=0.01):
    """
    Check if training has converged based on relative standard deviation of recent losses.

    Computes the coefficient of variation (std/mean) over a sliding window of recent
    loss values. Convergence is detected when this relative variability falls below
    a threshold.

    Inputs:
    -------
    loss_list : list
        List of loss values from training (one per epoch/iteration).
    window : int, default 5
        Number of recent epochs to analyze for convergence.
    std_threshold : float, default 0.01
        Threshold for relative standard deviation (std/mean). Training is considered
        converged if std/mean < std_threshold.

    Returns:
    --------
    converged : bool
        True if training has converged (recent losses are stable), False otherwise.
        Returns False if loss_list has fewer than 'window' elements.
    """
    if len(loss_list) < window:
        return False  # Not enough data

    recent_losses = loss_list[-window:]
    std_dev = np.std(recent_losses)
    mean = np.mean(recent_losses)
    std_dev_rel = std_dev / mean

    return std_dev_rel < std_threshold  # Converged if std deviation is very small
