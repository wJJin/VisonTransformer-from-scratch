import torch
import torch.nn as nn


def patchify(batch_images:torch.Tensor, patch_side:int) -> torch.Tensor:
    """
    Patchify the images.
    Shape:
        batch_images: (b, h, w, c) =>
        batch_patches: (b, n, p^2*c)
    """
    
    b, h, w, c = batch_images.shape
    p = patch_side
    batch_patches = torch.zeros(b, h*w//(p**2), (p**2)*c)
    
    for idx, batch_image in enumerate(batch_images):
        for i in range(h//p):
            for j in range(w//p):
                patch = batch_image[i*p : (i+1)*p, j*p : (j+1)*p, :]
                batch_patches[idx, j+h//p*i] = patch.flatten()
                
    return batch_patches


class PatchEmbeddings(nn.Module):
    """
    Flatten patches are passed through a feed-forward layer for linear projection,
    followed by the addition of positional embeddings.
    """

    def __init__(self, batch_shape:tuple, patch_side:int, hidden_dim:int, dropout:float):
        
        b, h, w, c = batch_shape
        super().__init__()
        
        self.p = patch_side
        self.projection = nn.Linear((self.p**2)*c, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(b, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(b, h*w//(self.p**2)+1, hidden_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = patchify(x, self.p)
        x = self.projection(x)
        x = torch.cat((self.cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        return x