import torch
import torch.nn as nn

class mlp_eq(nn.Module):
    def __init__(self, args):
        super(mlp_eq, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), # optional
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(self, x):
        out = self.model(x)
        
        return out