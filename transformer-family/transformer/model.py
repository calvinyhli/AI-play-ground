"""
@author:liyinghao
@homepage:https://github.com/calvinyhli
"""

import torch
from torch import nn
from transformer.blocks.encoder_layer import EncoderLayer
from transformer.blocks.decoder_layer import DecoderLayer
from transformer.embedding.positional_embedding import PostionalEmbedding
from transformer.embedding.token_embeddings import TokenEmbedding

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_hidden, n_head, n_layers, drop_prob) -> None:
        super().__init__()
        self.pos_emb = PostionalEmbedding(d_model, max_len)
        self.token_emb = TokenEmbedding(vocab_size, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_head, drop_prob) for _ in range(n_layers)])
    
    def forward(self, x, mask=None):
        x = self.token_emb(x) + self.pos_emb(x) 

        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_hidden, n_head, n_layers, drop_prob) -> None:
        super().__init__()
        self.pos_emb = PostionalEmbedding(d_model, max_len)
        self.token_emb = TokenEmbedding(vocab_size, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, d_hidden, n_head, drop_prob) for _ in range(n_layers)])

        self.out_linear = nn.Linear(d_model, vocab_size)

    def forward(self, de_in, en_in, trg_mask=None, src_mask=None):
        x = self.token_emb(de_in) + self.pos_emb(de_in)
        
        for layer in self.layers:
            x = layer(x, en_in, trg_mask, src_mask)

        output = self.out_linear(x)

        return output

class transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, vocab_size, max_len, d_model, d_hidden, n_head, n_layers, drop_prob) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.Encoder = Encoder(vocab_size=vocab_size,
                               max_len=max_len,
                               d_model=d_model,
                               d_hidden=d_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob
                               )
        self.Decoder = Decoder(vocab_size=vocab_size,
                               max_len=max_len,
                               d_model=d_model,
                               d_hidden=d_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               drop_prob=drop_prob
                               )
    def src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        print(src_mask)
        return src_mask
    
    def trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        seq_len = trg.shape[-1]
        trg_sub_mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.int)
        trg_mask = trg_pad_mask & trg_sub_mask
        print(trg_mask)
        return trg_mask
    
    def forward(self, src, trg):

        x = self.Encoder(src, self.src_mask(src))
        output = self.Decoder(trg, x, self.trg_mask(trg), self.src_mask(src))
        
        return output

if __name__ == "__main__":
    net = transformer(src_pad_idx=1,
                      trg_pad_idx=1,
                      vocab_size=128, 
                      max_len=64, 
                      d_model=8, 
                      d_hidden=16, 
                      n_head=2, 
                      n_layers=1, 
                      drop_prob=0.1)
    
    src = torch.randint(low=0, high=128, size=(4,32))
    trg = torch.randint(low=0, high=128, size=(4,32))
    print(src)

    output = net(src, trg)

    print(output)
    print(output.shape)