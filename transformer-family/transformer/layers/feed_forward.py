import torch
from torch import nn

class FeedForward(nn.Module):
    """
    feed forward implementation by hand
    """
    def __init__(self, d_model, hidden, drop_prob=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.linear2(x)
        return out
    
if __name__ == "__main__":
    d_model = 16
    x = torch.rand((2, 4, 16))
    layer = FeedForward(d_model, 32)
    output = layer(x)

    print(output)
    print(output.shape)





