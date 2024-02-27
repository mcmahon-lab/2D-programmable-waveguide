import wandb
import pytorch_lightning as pl
import os

def CustomCheckpoint(log_dir):
    checkpoint_file = os.path.join(log_dir, "{epoch}-{val_loss:.5f}")
    checkpoint_cb = pl.callbacks.ModelCheckpoint(checkpoint_file)
    return checkpoint_cb

def get_log_dir(pname, name):
    return os.path.join("logs", pname, name)

from pytorch_lightning.callbacks import Callback


def get_dirname(): 
    if os.name == "nt":
        path = os.getcwd()
        current_dir = os.path.basename(path)
    return current_dir

def get_logger(pname, name, team_name = 'pnn'):
    """
    Return pytorch lightning loggers.
    pname refers to the "project name" (which hosts multiple runs), while name refers to the name of the run
    In particular, it returns both the csv and wandb loggers that will output the log of both files.
    The wandb name will append the dirname to make things more clear.
    
    Also the "finish" command will make things a bit unclean, will figure out how to supress output later
    """
    wandb.finish(quiet=True) #To terminate any past runs!
    csv_logger = pl.loggers.CSVLogger('logs', pname, name)
    wandb_pname = wandb_pname = f"{get_dirname()}--{pname}"  # the name that will appear in wandb
    wandb_logger = pl.loggers.WandbLogger(name=name, project=wandb_pname, entity = team_name)
    logger = [csv_logger, wandb_logger]
    return logger