import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    """
    Residual connection for training the neural networks.
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        
    def forward(self, x):
        res = x
        x = self.layer(x)
        x += res
        
        return x
    

class AttentionHead(nn.Module):
    """
    A single attention head for MultiHeadAttention.
    This module computes attention scores for different aspects of the input data and is a crucial component of multi-head attention.
    """
    def __init__(self, hidden_dim, num_heads, dropout, bias=True):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=bias) #wQ
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=bias) #wK
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=bias) #wV
        self.dropout = nn.Dropout(dropout)
        self.scaling = (hidden_dim//num_heads)**(-0.5)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        x = torch.matmul(query, key.transpose(-1, -2)) / self.scaling
        x = nn.functional.softmax(x, dim=-1)
        x = self.dropout(x)
        x = torch.matmul(x, value)
        
        return x
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    This module combines the results from multiple attention heads to capture diverse features in the input data.
    """
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList([])
        for _ in range(num_heads):
            head = AttentionHead(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.heads.append(head)
            
        self.projection = nn.Linear(hidden_dim*num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = [head(x) for head in self.heads]
        x = torch.cat([out for out in x], dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        
        return x
    

class TransformerEncoderBlock(nn.Sequential):
    """
    A single block within the Transformer encoder.
    This block applies multi-head attention and feed-forward neural networks to process and refine the input data.
    """
    def __init__(self, hidden_dim, batch_shape, patch_side, dropout, num_heads):
        
        b, h, w, c = batch_shape
        in_shape = h * w // (patch_side ** 2) + 1
        
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(torch.Size([in_shape, hidden_dim])),
                MultiHeadAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                ),
                nn.Dropout(dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(torch.Size([in_shape, hidden_dim])),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            ))


class TransformerEncoder(nn.Sequential):
    """
    The main Transformer encoder composed of multiple TransformerEncoderBlocks.
    This encoder stacks multiple TransformerEncoderBlocks to process and transform the input data.
    """
    def __init__(self,
                 depth: int,
                 hidden_dim: int,
                 batch_shape: tuple,
                 patch_side: int,
                 dropout: float,
                 num_heads:int):
        super().__init__(*[
            TransformerEncoderBlock(hidden_dim=hidden_dim, batch_shape=batch_shape, patch_side=patch_side, dropout=dropout, num_heads=num_heads) for _ in range(depth)
        ])
