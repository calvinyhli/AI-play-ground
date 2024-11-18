import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    layernorm layer implementation by hand
    """
    def __init__(self, d_model, eps=1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var+self.eps)

        return self.gamma*out + self.beta
    
if __name__ == "__main__":
    d_model = 16
    x = torch.rand((2, 4, 16))
    layernorm = LayerNorm(x.size()[-1])
    layernorm_torch = nn.LayerNorm(d_model, eps=1e-12)
    layernorm_torch.weight.data.copy_(layernorm.gamma.data)
    layernorm_torch.bias.data.copy_(layernorm.beta.data)

    output = layernorm(x)
    output_torch = layernorm_torch(x)

    comparison = torch.allclose(output_torch, output, atol=1e-6)

    print(output)
    print(output.shape)
    print("Are outputs close? ", comparison)





