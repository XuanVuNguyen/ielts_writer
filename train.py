import argparse
import yaml
import torch
import pytorch_lightning as pl
from model.data import IelstDataset, train_val_split
from model.config import get_train_config
from model.model import GPT2Lightning
from model.train import train

def setup_arg_parser():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument("--config", type=str, 
                        default="config/config.yaml")    
    return parser

def main(args):
    torch.cuda.empty_cache()
    pl.seed_everything(0)
    
    with open(args.config, 'r') as file:
        config_yaml = yaml.load(file, Loader=yaml.FullLoader)
    config = get_train_config(config_yaml)
    print(config)
    model = GPT2Lightning(config)
    
    dataset = IelstDataset(model.config.data_path,
                           max_length=model.config.max_length, padding=True)
    assert model.gpt2.get_input_embeddings().num_embeddings == len(dataset.tokenizer)
    train_set, val_set = train_val_split(dataset, 0.9)
    
    train(model, train_set, val_set)
    
if __name__=="__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    main(args)