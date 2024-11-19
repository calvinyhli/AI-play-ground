import torch
from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model) -> None:
        """
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super().__init__(vocab_size, d_model, padding_idx=1)

if __name__ == "__main__":
    x = torch.randint(low=1,high=100, size=(4,16))
    token_emb = TokenEmbedding(vocab_size=128, d_model=32)
    out = token_emb(x)
    print(out)
    print(out.shape)