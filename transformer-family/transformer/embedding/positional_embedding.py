import torch
from torch import nn

class PostionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len) -> None:
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :]
    
if __name__ == "__main__":
    x = torch.rand(2, 4, 16)
    embedding = PostionalEmbedding(d_model=x.size()[-1], max_len=1024)
    out = embedding(x)
    print(x)
    print(out)
    print(out.shape)

 
         

    
