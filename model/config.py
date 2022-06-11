from typing import Optional, Tuple, Union
from dataclasses import dataclass

@dataclass(repr=True)
class Config:
    pretrained_name_or_path: str = "./pretrained/gpt2"
    batch_size: int = 2
    max_length: int = 500
    num_training_epochs: int = 100
    num_warmup_epochs: int = 2
    lr: float = 5e-4
    
    # log_frequency: int = 2
    eval_frequency: int = 2
    
    device_name: Optional[str] = "gpu"
    devices: Union[int, Tuple[int]] = (1, )
    grad_accumulate_steps: int = 8
    grad_checkpointing: bool = False
    parallel_strategy: str = "ddp"
    
    data_path: str = "data/data.csv"
    log_dir: str = "logs"
    exp_name: str = "test"    

def get_train_config(config: Optional[dict]=None):
    if config is None:
        config = {}
    output_config = Config()
    for key, value in config.items():
        if key in output_config.__dict__.keys():
            if isinstance(value, list):
                value = tuple(value)
            setattr(output_config, key, value)
    
    return output_config