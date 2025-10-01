import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def L2(p, q):
    # L2 distance between p and q
    x = torch.abs(p-q)**2
    return x.sum(dim = -1).mean()/2

def relu_approx(x, factor=20.0):
    """
    A soft-relu function.
    https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    
    Parameters:
    -----------
    factor : float, default: 20.
        Factor that determines shape of softplus. The larger, the closer to relu, the smaller, the smoother.
    """
    return F.softplus(x*factor)/factor

def clamp_lag(x, low=0.0, high=1.0, factor=20.): 
    """
    Returns loss that is higher the more the values of x exceed the lower and upper threshold.
    
    Approximately returns mean distance of the elements of x from the interval [low, high]. The
    higher factor, the more accurate the returned value is to the mean distance.
    
    Parameters:
    -----------
    x : torch.tensor
        Tensor for which loss is calculated.
    low : float, default: 0.0
        Lower boundary. Any values of x that are lower than low by a value of delta acquire a loss of
        approximately delta.
    high: float, default: 1.0
        Upper boundary. Any values of x that are higher than high by a value of delta acquire a loss of
        approximately delta.
    factor: float, default: 20.0
        Factor determining exact shape of loss function. The larger factor the closer the loss follows 
        a ReLu function.
    """
    return torch.mean(relu_approx(-(x-low), factor) + relu_approx(x-high, factor)) 


def lagrangian(model, lag_amp = 1., factor=20.0):
    """
    The lagrangian function that can be added to the loss during a training loop.

    A softened ReLu function is applied to each input. The loss is applied in normalized inputs
    where the lower bound is always 0 and the upper bound always 1. For each distance delta that
    the inputs are outside of that range, approximately delta*lag_amp loss is added.

    Parameters:
    -----------
    lag_amp : float, default: 1.
        Multiplier that determines how much loss is added for inputs outside of input_range.

    Outputs:
    -----------
    loss : torch.tensor
        Tensor containing the value of the added loss.
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
    Subclass of nn.Parameter that additionally stores 
    an upper and lower bound for each parameter.
    """
    def __new__(cls, data=None, requires_grad=True, limits=None):
        """
        Parameter:
        -----------
        data : torch.tensor
            Tensor containing the values of the parameter.
        requires_grad : bool
            Determines whether computation tree is built for par
        limits : list of floats
            Specifies the lower and upper bound of the parameter that can be
            used during PNN training to keep the parameter within those limits.
        """
        param = nn.Parameter.__new__(cls, data = data, requires_grad=requires_grad)
        param.limits = limits
        return param
    
    def __repr__(self):
        return 'Parameter containing:\n' + super(nn.Parameter, self).__repr__() + '\tLimits: ' + str(self.limits)
        
def has_converged(loss_list, window=5, std_threshold=0.01):
    """
    Check if training has converged based on the standard deviation of recent loss values.

    Args:
        loss_list (list): List of loss values per epoch.
        window (int): Number of recent epochs to consider.
        std_threshold (float): Maximum acceptable standard deviation.

    Returns:
        bool: True if loss has converged, False otherwise.
    """
    if len(loss_list) < window:
        return False  # Not enough data

    recent_losses = loss_list[-window:]
    std_dev = np.std(recent_losses)
    mean = np.mean(recent_losses)
    std_dev_rel = std_dev / mean

    return std_dev_rel < std_threshold  # Converged if std deviation is very small
