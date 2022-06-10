from typing import Optional, Union, List
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def collate_fn(batch_items):
        input_id_tensors = [item[0] for item in batch_items]
        attention_mask_tensors = [item[1] for item in batch_items]
        texts = [item[2] for item in batch_items]
        return torch.cat(input_id_tensors), torch.cat(attention_mask_tensors), texts
    
def get_train_val_dataloaders(train_set: Dataset, val_set: Dataset, batch_size=32):    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True)
    
    return train_loader, val_loader

def train(model: pl.LightningModule,
          train_set: Dataset,
          val_set: Dataset):

    if model.config.device_name is None:
        device_name = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        device_name = model.config.device_name
        
    train_loader, val_loader = get_train_val_dataloaders(train_set, val_set, batch_size=model.config.batch_size)
    
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        monitor="eval/log_perplexity",
        # every_n_train_steps=model.config.eval_frequency,
        every_n_epochs=model.config.eval_frequency
    )
    # eval_callback = EvalCallback(model.config.eval_frequency, val_loader)
    lr_monitor = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(
        save_dir = model.config.log_dir,
        name=model.config.exp_name,
        log_graph=False,
        default_hp_metric=False
    )
    trainer = pl.Trainer(max_epochs=model.config.num_training_epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         check_val_every_n_epoch=model.config.eval_frequency,
                         logger=logger,
                         accelerator=device_name,
                         devices=model.config.devices,
                         strategy=model.config.parallel_strategy,
                        #  precision=16,
                         accumulate_grad_batches=model.config.grad_accumulate_steps)
        
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)