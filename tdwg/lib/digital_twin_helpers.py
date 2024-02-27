import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import pytorch_lightning as pl

def np2loaders(x, y, train_ratio=0.9, Nbatch = 100, dtype = torch.float32, num_workers = 0):
    """
    Produces the training and testing pytorch dataloaders given numpy inputs
    x: input data in numpy format. First dimension is batch dimension
    y: output data in numpy format. First dimension is batch dimension
    train_ratio: The ratio of dataset that will be training data
    Nbatch: batchsize for the training set
    Note the testset has a batchsize of the whole training set
    """
    Ntotal = x.shape[0]
    Ntrain = int(np.floor(Ntotal*train_ratio))
    train_inds = np.arange(Ntrain)
    val_inds = np.arange(Ntrain, Ntotal)
    X_train = torch.tensor(x[train_inds], dtype = dtype)
    X_val = torch.tensor(x[val_inds], dtype = dtype)
    Y_train = torch.tensor(y[train_inds], dtype = dtype)
    Y_val = torch.tensor(y[val_inds], dtype = dtype)
    train_dataset = data.TensorDataset(X_train, Y_train)
    val_dataset = data.TensorDataset(X_val, Y_val)
    train_loader = data.DataLoader(train_dataset, Nbatch, num_workers = num_workers)
    val_loader = data.DataLoader(val_dataset, val_dataset.tensors[0].shape[0], num_workers = num_workers)
    return train_loader, val_loader

def swish(x):
    return x * torch.sigmoid(x)

class Attenuator(nn.Module):
    """
    Initializes mostly in the identity map
    """
    def __init__(self, dim, factor_init=0.8):
        super().__init__()
        self.factors = nn.Parameter(factor_init*torch.ones(dim))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factors
        return self.out
    
    
class LitUnitaryWithoutOutputFacet(pl.LightningModule):
    def __init__(self, input_dim, output_dim, Nunits, dtype = torch.float32):
        super().__init__()
        self.layers = []
        if dtype == torch.complex32 or dtype == torch.complex64:
            self.loss_f = lambda x, y: F.mse_loss(torch.abs(x), torch.abs(y))
        else: 
            self.loss_f = F.mse_loss
            
        self.input_dim = input_dim
        self.lr = 5e-2
        self.step_size = 20

        self.orth_linear = nn.utils.parametrizations.orthogonal(nn.Linear(input_dim, output_dim, dtype = torch.complex64))
        self.input_facet = Attenuator(input_dim, 1)
            
    def forward(self, data):
        data = self.input_facet(data)
        data = self.orth_linear(data)
        return data.abs()**2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)    
        loss = self.loss_f(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)    
        loss = self.loss_f(y_hat, y)
        self.log('val_loss', loss)
        
class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, Nunits, dtype = torch.float32):
        super().__init__()
        self.layers = []
        if dtype == torch.complex32 or dtype == torch.complex64:
            self.loss_f = lambda x, y: F.mse_loss(torch.abs(x), torch.abs(y))
        else: 
            self.loss_f = F.mse_loss
            
        self.lr = 5e-2
        self.step_size = 20

        for Nunit in Nunits:
            self.layers.append(nn.Linear(input_dim, Nunit))
            input_dim = Nunit
        
        self.layers.append(nn.Linear(input_dim, output_dim, dtype = dtype, bias = False))

        # Assigning the layers as class variables (PyTorch requirement). 
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)

    def forward(self, data):
        for layer in self.layers[:-1]:
            data = layer(data)
            data = swish(data)
        data = self.layers[-1](data)
        return data.abs()**2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), "monitor": "train_loss"}
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)    
        loss = self.loss_f(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)    
        loss = self.loss_f(y_hat, y)
        self.log('val_loss', loss)
