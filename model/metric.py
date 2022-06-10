import torch
import torch.nn.functional as F

def log_perplexity(input_ids: torch.Tensor, attention_mask: torch.Tensor, logits: torch.Tensor):
    input_ids = input_ids[:, 1:]
    attention_mask = attention_mask[:, 1:]
    probs = F.softmax(logits, dim=-1)
    input_token_probs = probs[torch.arange(probs.size(0)).unsqueeze(-1), torch.arange(logits.size(1)-1), input_ids]
    input_token_probs = input_token_probs ** attention_mask
    input_probs = input_token_probs.prod(-1)
    log_p = -torch.log(input_probs) / attention_mask.sum(-1)
    log_p = log_p.mean()
    return log_p