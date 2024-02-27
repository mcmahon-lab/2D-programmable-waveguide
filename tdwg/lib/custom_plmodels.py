import torch
import torch.nn as nn
import numpy as np
from .pl_utils import *
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader

class LagrangePlModel(pl.LightningModule):
    """
    Goals of this code:
    1. Take care of all of the boiler plate training loop
    2. Add in the lagragian term to the training
    3. Having saving built into the repo
    """
    def __init__(
        self, train_dataset, val_dataset, test_dataset, pnn, lag_func,
        learning_rate=2e-4,  gamma = 0.98, batch_size = 64, 
        save_outputs_every_n_epochs = None):
        super().__init__()

        # Set our init args as class attributes
        self.save_hyperparameters( ignore=["train_dataset", "val_dataset",
                            "test_dataset", "pnn", "lag_func"])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.pnn = pnn
        self.lag_func = lag_func
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.val_batch_size = 100

        self.metric = Accuracy(task="multiclass", num_classes = len(train_dataset.classes))
        
        self.save_outputs_every_n_epochs = save_outputs_every_n_epochs
        self.validation_outputs = []
        
    def forward(self, x):
        logits = self.pnn.forward(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, target = batch
        logits = self.pnn.forward(x)
        l_objective = F.nll_loss(logits, target)
        l_lagrange = self.lag_func(self.pnn)
        loss = l_objective + l_lagrange 
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, target)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_obj", l_objective, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_lag", l_lagrange, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        logits = self.pnn.forward(x)
        loss = F.nll_loss(logits, target)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, target)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        #At the very least, if I save this, I will be able to make confusion matrices!
        save_dict = {"val_x": x, "val_logits" : logits, "val_target" : target, "val_preds" : preds, 
                     "val_loss" : loss, "val_acc" : acc, "batch_idx":batch_idx,
                     "epoch":self.current_epoch, "state_dict":self.state_dict()}

        if batch_idx < 10: #or some number, save the trajectories and stuff, I guess
            if hasattr(self.pnn, "save_dict"):
                save_dict.update(self.pnn.save_dict) #add as many variables as warranted, as given by this...
        
        self.validation_outputs.append(save_dict)

    def on_validation_epoch_end(self):
        """
        Run the save every few epochs. The code as written, saves everything though...
        #Ok, this is a reasonable thing to discuss with Martin tomorrow -> Need to improve the saving somewhat, so that we can handle MNIST... With thousands of files, it will affect MNIST, so we need to deal with the situation
        """
        if self.save_outputs_every_n_epochs:
            if self.current_epoch == 0 or (self.current_epoch+1) % self.save_outputs_every_n_epochs == 0 or (self.current_epoch+1) == self.trainer.max_epochs:
                try:
                    save_dir = self.trainer.logger.log_dir + f'\\epoch{self.current_epoch}_val_data.pth'
                except:
                    log_dir = self.trainer.logger[0].log_dir
                    save_dir = self.trainer.log_dir + '\\' + self.trainer.logger[0].log_dir + f'\\epoch{self.current_epoch}_val_data.pth'
        
                torch.save(self.validation_outputs, save_dir)

        #I see so it always deletes this AFTER each epoch!
        self.validation_outputs = []

    def test_step(self, batch, batch_idx):
        x, target = batch
        logits = self.pnn.forward(x)
        loss = F.nll_loss(logits, target)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, target)
        acc = self.metric(preds, target)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.gamma),
                "interval": "epoch", "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size) #can increase this to reduce the number that I save? 

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size)


############# Simple objects! (From the PNN project) ##############

class Manifold(nn.Module):
    """
    Initializes mostly in the identity map
    """
    def __init__(self, dim, factor_init=0.8, offset_init=0.2):
        super().__init__()
        self.factors = nn.Parameter(factor_init*torch.ones(dim))
        self.offsets = nn.Parameter(offset_init*torch.zeros(dim))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factors + self.offsets
        self.save_dict = dict(inp=x.detach(), out=self.out.detach())
        return self.out
    
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

        self.save_dict = dict(inp=x.detach(), out=self.out.detach())
        return self.out
    
class SingleManifold(nn.Module):
    """
    Initializes mostly in the identity map
    """
    def __init__(self, factor_init=1.0, offset_init=0.0):
        super().__init__()
        self.factor = nn.Parameter(factor_init*torch.tensor(factor_init))
        self.offset = nn.Parameter(offset_init*torch.tensor(offset_init))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factor + self.offset

        self.save_dict = dict(inp=x.detach(), out=self.out.detach())
        return self.out
    
class SingleMult(nn.Module):
    def __init__(self, factor_init=1.0):
        super().__init__()
        self.factor = nn.Parameter(factor_init*torch.tensor(factor_init))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factor 

        self.save_dict = dict(inp=x.detach(), out=self.out.detach())
        
        return self.out

class Skip(nn.Module):
    """
    This skip is "wierd" in that it adds a weight on the skip side.
    x_skip in code refers to neuron that is far away (being skipped forward)
    """
    def __init__(self, skip_weight=1.0):
        super().__init__()
        self.skip_weight = nn.Parameter(torch.tensor(skip_weight))
        
    def forward(self, x, x_skip):
        out = x + x_skip*self.skip_weight #normal skips would do x_skip + x*x_weight...
        self.save_dict = dict(inp=x.detach(), inp_skip=x_skip.detach(), out=out.detach())
        return out
