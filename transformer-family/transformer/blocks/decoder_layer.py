import torch
from torch import nn

from transformer.layers.feed_forward import FeedForward
from transformer.layers.layernorm import LayerNorm
from transformer.layers.multi_head_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, drop_prob=0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.feedforward = FeedForward(d_model, d_hidden, drop_prob)
        self.layernorm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
    
    def forward(self, dec, enc, trg_mask=None, src_mask=None):
        # 1.compute self attention
        _x = dec # for skip connection
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2.add & norm
        x = self.dropout1(x)
        x = self.layernorm1(x + _x)

        # 3.compue encoder-decoder attention
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.layernorm2(x + _x)

        # 3.feed forward
        _x = x
        x = self.feedforward(x)

        # 4.add & norm
        x = self.dropout3(x)
        x = self.layernorm3(x + _x)

        return x

if __name__ == "__main__":
    enc = torch.rand((2, 4, 8))
    dec = torch.rand((2, 4, 8))

    Decoder = DecoderLayer(8, 16, 2)
    out = Decoder(dec, enc)
    print(out)
    print(out.shape)




