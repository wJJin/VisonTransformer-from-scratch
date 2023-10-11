import torch.nn as nn

class MLPHead(nn.Module):
    """
    MLP for classification.
    The model for pretrained has 1 hidden layer.
    """
    def __init__(self, hidden_dim:int, classes:int, dropout:float):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim//2) #wQ
        self.output_layer = nn.Linear(hidden_dim//2, classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x[:,0,:]
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        x = self.activation(x)
        
        return x