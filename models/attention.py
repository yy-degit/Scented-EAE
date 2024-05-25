import torch
from torch import nn
import math
import torch.nn.functional as F
    

def one_head_attention(query, key, value, k_mask, dropout):
    '''
    Args:
        query: [batch_size x q_length x hidden_size]
        key: [batch_size x k_length x hidden_size]
        value: [batch_size x v_length x hidden_size]
        k_mask: [batch_size  x k_length]
        mask is 0 if it is masked
    Returns:
        output: [batch_size x q_length x hidden_size]
    '''
    q_length= query.size(1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    mask = k_mask.unsqueeze(-2).expand(-1, q_length, -1)
    mask = (1.0-mask) * (-10000.0)
    scores = scores + mask
    qk_attention = F.softmax(scores, dim=-1)
    qk_attention = dropout(qk_attention)
    output = torch.matmul(qk_attention, value)
    return output