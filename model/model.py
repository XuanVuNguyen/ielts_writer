import torch
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import get_linear_schedule_with_warmup

# from slide2note.data import tokenizer
from model.config import Config
from model.metric import log_perplexity
from fairscale.nn import checkpoint_wrapper

class GPT2Lightning(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._n_devices = len(config.devices) if isinstance(config.devices, tuple) else config.devices
        self.gpt2_config = GPT2Config.from_pretrained("gpt2")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.pretrained_name_or_path, config=self.gpt2_config)
        self.gpt2.resize_token_embeddings(self.gpt2.get_input_embeddings().num_embeddings + 2)
        if config.grad_checkpointing:
            self.gpt2 = checkpoint_wrapper(self.gpt2)
        self.save_hyperparameters()
        
    def forward(self, batch_item: tuple):
        input_ids = batch_item[0]
        labels = input_ids.clone()
        attention_mask = batch_item[1]
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def training_step(self, batch_item: tuple, batch_idx):
        outputs = self(batch_item)
        # if self.global_step // self._n_devices % self.config.log_frequency == 0:
        self.log("train/loss", 
                 outputs.loss, 
                 sync_dist=True,
                 on_step=True, 
                 on_epoch=True,
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.config.batch_size*self._n_devices)
        return outputs.loss
    
    def training_epoch_end(self, train_step_outputs):
        loss = train_step_outputs[-1]["loss"].item()
        # self.log("train/loss", loss, sync_dist=True)
        
        print(f"\n>>>Epoch: {self.current_epoch}; Global step: {self.global_step}; Loss: {loss}")
    
    def validation_step(self, batch_item: tuple, batch_idx):
        outputs = self(batch_item)
        input_ids, attention_mask, _ = batch_item
        logits = outputs.logits
        log_p = log_perplexity(input_ids, attention_mask, logits)
        self.log("eval/loss", 
                 outputs.loss, 
                 sync_dist=True,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.config.batch_size*self._n_devices)
        
        self.log("eval/log_perplexity", 
                 log_p, 
                 sync_dist=True,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.config.batch_size*self._n_devices)
        return outputs.loss.item()
    
    # def validation_epoch_end(self, validation_step_outputs):
    #     # losses = [output["loss"] for output in validation_step_outputs]
    #     losses = validation_step_outputs
    #     eval_loss = sum(losses)/len(losses)
    #     self.log("eval/loss", eval_loss, sync_dist=True, logger=True)
    #     print(f"\n>>>Validation loss: {eval_loss}")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps = self.config.num_warmup_epochs,
            num_training_steps = self.config.num_training_epochs)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}