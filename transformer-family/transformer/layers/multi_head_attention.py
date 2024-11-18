import math

import torch
from torch import nn
import torch.nn.functional as F

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    """
    def __init__(self) :
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # input:[batch_size, n_head, length, d_hidden]
        batch_size, n_head, length, d_hidden = q.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_hidden) # scale dot-product

        # apply masking (opt)
        if mask is not None:
            # print("mask is:", mask)
            score = score.masked_fill(mask==0, float('-inf'))
            # print("score is:",score)
        
        # compute final score after softmax
        score = F.softmax(score, dim=-1)
        v = score @ v

        return v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head) -> None:
        super().__init__()
        self.attention = ScaleDotProductAttention()
        self.d_model = d_model
        self.d_head = d_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split tensor for multi-head
        batch_size, length, d_model = q.size()
        d_hidden = d_model // self.d_head
        q = q.view(batch_size, length, self.d_head, d_hidden).transpose(1,2) # b, l, h ==> b, n, l, h'
        k = k.view(batch_size, length, self.d_head, d_hidden).transpose(1,2)
        v = v.view(batch_size, length, self.d_head, d_hidden).transpose(1,2)

        # do scale dot product attention
        out = self.attention(q, k, v, mask=None)

        # concat
        out = out.transpose(1, 2).contiguous().view(batch_size, length, d_model) # b, n, l, h' ==> b, l, h

        out = self.w_o(out)

        return out
    
if __name__ == "__main__":
    q = torch.rand((2, 4, 16))
    k = torch.rand((2, 4, 16))
    v = torch.rand((2, 4, 16))
    attention = MultiHeadAttention(q.size()[-1], 2)
    output = attention(q, k, v)
    print(output)
    print(output.shape)





