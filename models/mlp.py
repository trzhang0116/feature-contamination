import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Linear):

    def __init__(self, input_dim, output_dim, weights=None, biases=None, normalize=False):
        super().__init__(input_dim, output_dim)

        self.normalize = normalize
        if weights is not None:
            self.weight = nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = nn.Parameter(biases.clone())
        else:
            self.bias = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)
    
    
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, reversible=False, residual=False, requires_grad=True):
        super().__init__()

        if reversible:
            assert hidden_dim == input_dim
            base_block = RevBlock
        elif residual:
            base_block = ResBlock
        else:
            base_block = BasicBlock
        
        if n_layers == 1:
            blocks = [base_block(input_dim, hidden_dim)]
        elif n_layers == 2:
            blocks = [base_block(input_dim, hidden_dim),
                      base_block(hidden_dim, hidden_dim)]
        else:
            assert n_layers >= 3
            blocks = [base_block(input_dim, hidden_dim)] + \
                     [base_block(hidden_dim, hidden_dim) for _ in range(n_layers-2)] + \
                     [base_block(hidden_dim, hidden_dim)]

        self.mlp = nn.ModuleList(blocks)
        self._set_requires_grad(requires_grad)

        self.reversible = reversible
    
    def forward(self, x):
        for block in self.mlp:
            x = block(x)
        return x
    
    def _set_requires_grad(self, status):
        for params in self.mlp.parameters():
            params.requires_grad = status


class BasicBlock(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.F = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.F(x)


class ResBlock(nn.Module):

    def __init__(self, input_dim, output_dim, final_relu=True):
        super().__init__()

        self.F = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.final_relu = final_relu
        
        if final_relu:
            self.final_bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        if self.final_relu:
            fx = self.final_bn(self.F(x))
            return F.relu(x + fx)
        else:
            return x + self.F(x)


class RevBlock(nn.Module):
    # Reversible residual block
    # https://arxiv.org/pdf/1707.04585.pdf
    def __init__(self, input_dim, output_dim):
        super().__init__()

        assert input_dim == output_dim
        assert input_dim % 2 == 0
        self.half_input_dim = input_dim // 2

        self.F = nn.Sequential(
            nn.Linear(self.half_input_dim, self.half_input_dim),
            nn.BatchNorm1d(self.half_input_dim),
            nn.ReLU(),
            nn.Linear(self.half_input_dim, self.half_input_dim)
        )
        self.G = nn.Sequential(
            nn.Linear(self.half_input_dim, self.half_input_dim),
            nn.BatchNorm1d(self.half_input_dim),
            nn.ReLU(),
            nn.Linear(self.half_input_dim, self.half_input_dim)
        )
    
    def forward(self, x):
        # Forward rule:
        # x1, x2 = split(x)
        # y1 = x1 + F(x2)
        # y2 = x2 + G(y1)
        # y = concat(y1, y2)
        x1, x2 = torch.split(x, self.half_input_dim, dim=1)
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], dim=1)
    
    def reconstruct(self, y):
        # Reconstructing rule:
        # y1, y2 = split(y)
        # x2 = y2 - G(y1)
        # x1 = y1 - F(x2)
        # x = concat(x1, x2)
        y1, y2 = torch.split(y, self.half_input_dim, dim=1)
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=1)
    