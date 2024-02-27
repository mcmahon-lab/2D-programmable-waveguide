import torch
import torch.nn as nn
import torch.nn.functional as F

def binarize(tensor, v1, v2):
    # binarize tensor to values v1 and v2
    # every element will be rounded to the value that it is closer to
    tensor = (tensor - v1) / (v2 - v1)
    tensor = torch.round(tensor)
    tensor = torch.clamp(tensor, min = 0, max = 1)
    return (v2 - v1) * tensor + v1

def EMD(p, q):
    # earth's mover distance between distributions p and q
    n = p.shape[-1]
    x = p - q
    y = torch.cumsum(x, dim=-1)/n
    return torch.abs(y).sum(dim=-1).mean()*2

def L1(p, q):
    # L1 distance between p and q
    x = torch.abs(p-q)
    return x.sum(dim = -1).mean()/2

def L2(p, q):
    # L2 distance between p and q
    x = torch.abs(p-q)**2
    return x.sum(dim = -1).mean()/2

def biasing(tensor, k = 0):
    # (Eq. 4) from https://arxiv.org/pdf/1709.08809.pdf
    return 1 / (1-2*k) * (tensor - 1/2) + 1/2

def neighbor_biasing(tensor, k, radius, k_avg, p):
    # (Eq. 5) from https://arxiv.org/pdf/1709.08809.pdf
    z_avg = convolve_with_circle(tdwg_pnn.wg, tensor, radius)
    h = 1/2 + k_avg * (1/2 - z_avg)**(2*p + 1)
    return 1 / (1-2*k) * (tensor - h) + 1/2

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