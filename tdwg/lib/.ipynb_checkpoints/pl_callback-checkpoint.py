import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from tdwg.lib.sweeps import *
from tdwg.lib.plot_utils import plot_grid
from tdwg.lib.misc_utils import *

__all__ = ['check_digital_twins', 'Check_DT_Callback']

def check_digital_twins(tdwg_exp, tdwg_sim, mod_sim_window_additional = None, close_fig=True):
    """
    close_fig = True means that plot won't be shown in jupyter notebook. Set as True on default since it will mostly be used in a callback.
    """
    custom_mkdir("calibration_fig/dt_match")
    device = tdwg_sim.modulation_efficiency.device

    tdwg_sim.to("cpu")
    x_axis = tdwg_sim.wg.x_axis
    data_dict = torch.load("test_dt_data.pth")
    
    Ein_random = data_dict["Ein_random"]
    Ein_mnist = data_dict["Ein_mnist"]
    mod_sim_window_test = data_dict["mod_sim_window_test"]

    loss_additional = 0
    if mod_sim_window_additional != None:
        y_sim_list = sweep_zip(tdwg_sim.forward, Ein_mnist, [mod_sim_window_additional], verbose=False)
        y_exp_list = sweep_zip(tdwg_exp.forward, Ein_mnist, [mod_sim_window_additional], verbose=False)
        
        loss_list = [torch.nn.functional.mse_loss(y_sim, y_exp).detach() 
                     for (y_sim, y_exp) in zip(y_sim_list, y_exp_list)]
        loss_additional = np.mean(loss_list)
        
        fig, ax = plot_grid(x_axis, y_sim_list[0].detach(), y_exp_list[0], xlim=0.6);
        plt.suptitle(f"additional_loss={loss_additional}")
        plt.savefig(f"calibration_fig/dt_match/additional_{timestring()}")
        if close_fig:
            plt.close()
    
    y_sim_list = sweep_zip(tdwg_sim.forward, Ein_random, mod_sim_window_test, verbose=False)
    y_exp_list = sweep_zip(tdwg_exp.forward, Ein_random, mod_sim_window_test, verbose=False)
    
    loss_list = [torch.nn.functional.mse_loss(y_sim, y_exp).detach() 
                 for (y_sim, y_exp) in zip(y_sim_list, y_exp_list)]
    loss_random = np.mean(loss_list)
    
    fig, ax = plot_grid(x_axis, y_sim_list[0].detach(), y_exp_list[0], xlim=0.6);
    plt.suptitle(f"random_loss={loss_random}")
    plt.savefig(f"calibration_fig/dt_match/random_{timestring()}")
    if close_fig:
        plt.close()
    
    y_sim_list = sweep_zip(tdwg_sim.forward, Ein_mnist, mod_sim_window_test, verbose=False)
    y_exp_list = sweep_zip(tdwg_exp.forward, Ein_mnist, mod_sim_window_test, verbose=False)
    
    loss_list = [torch.nn.functional.mse_loss(y_sim, y_exp).detach() 
                 for (y_sim, y_exp) in zip(y_sim_list, y_exp_list)]
    loss_mnist = np.mean(loss_list)
    
    fig, ax = plot_grid(x_axis, y_sim_list[0].detach(), y_exp_list[0], xlim=0.6);
    plt.suptitle(f"mnist_loss={loss_mnist}")
    plt.savefig(f"calibration_fig/dt_match/mnist_{timestring()}")
    if close_fig:
        plt.close()

    tdwg_sim.to(device) #reset it back!
    return loss_random, loss_mnist, loss_additional

class Check_DT_Callback(pl.Callback):
    def __init__(self, tdwg_exp, tdwg_sim, every_n_steps):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.tdwg_exp = tdwg_exp
        self.tdwg_sim = tdwg_sim

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Check if the current step is a multiple of N
        # if (trainer.global_step + 1) % self.every_n_steps == 0:
        if (trainer.global_step) % self.every_n_steps == 0:
            loss_random, loss_mnist, loss_additional = check_digital_twins(
                self.tdwg_exp, self.tdwg_sim, trainer.model.pnn.tdwg_layer.tdwg.mod_sim_window.detach().cpu())
            
            pl_module.log('loss_random', loss_random, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            pl_module.log('loss_mnist', loss_mnist, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            pl_module.log('loss_current', loss_additional, on_step=True, on_epoch=False, prog_bar=True, logger=True)