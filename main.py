import torch
import numpy as np
import torch.nn as nn
from PIL import Image

import config

from patch_embedding import *
from transformer_encoder import *
from mlp_head import *


class VisionTransformer(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__(
            PatchEmbeddings(
                batch_shape=config.VIT_SETTINGS['input_shape'],
                patch_side=config.VIT_SETTINGS['patch_length'],
                hidden_dim=config.VIT_SETTINGS['dim_size'],
                dropout=config.VIT_SETTINGS['dropout_probs'],
            ),
            TransformerEncoder(
                depth=config.VIT_SETTINGS['encoder_nums'],
                hidden_dim=config.VIT_SETTINGS['dim_size'],
                batch_shape=config.VIT_SETTINGS['input_shape'],
                patch_side=config.VIT_SETTINGS['patch_length'],
                dropout=config.VIT_SETTINGS['dropout_probs'],
                num_heads=config.VIT_SETTINGS['head_nums'],
            ),
            MLPHead(
                hidden_dim=config.VIT_SETTINGS['dim_size'],
                classes=config.VIT_SETTINGS['classes'],
                dropout=config.VIT_SETTINGS['dropout_probs']
            ),
        )

def main():
    model = VisionTransformer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #one step for sample training:
    batch_images = torch.Tensor(np.array(Image.open('./sample.png'))).unsqueeze(0)
    output = model(batch_images)

if __name__ == '__main__':
    main()