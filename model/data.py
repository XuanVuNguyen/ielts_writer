from typing import Optional, List, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torch import Generator
from transformers import GPT2Tokenizer

class IelstDataset(Dataset):
    q_tag = "<|question|>"
    a_tag = "<|answer|>"
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        eos_token="<|endoftext|>", 
        pad_token="<|endoftext|>",
        additional_special_tokens=[q_tag, a_tag]
        )
    def __init__(
        self, 
        csv_path: str, 
        max_length: Optional[int]=None, 
        padding: bool=True):
        
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.padding = "max_length" if padding else False

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, id):
        row = self.df.iloc[id]
        question = row["question"]
        answer = row["answer"]
        text = self.q_tag + question + self.a_tag + answer + "<|endoftext|>"
        encode_dict = self.tokenizer(text, padding=self.padding, truncation=True, max_length=self.max_length, return_tensors="pt")
        self.tokenizer(text, padding=self.padding, truncation=True, max_length=self.max_length, return_tensors="pt")
        return encode_dict["input_ids"], encode_dict["attention_mask"], text

def train_val_split(dataset: Dataset, train_ratio: float=0.85, seed=0):
    train_len = int(len(dataset) *train_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=Generator().manual_seed(seed))
    return train_set, val_set