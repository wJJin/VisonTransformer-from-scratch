VIT_SETTINGS = {
    'patch_length': 16, #side length of token patches
    'dim_size': 2048, #constant latent vector size for hidden dimensions
    'input_shape': (1, 224, 224, 3), #batch_size, width, height, channel
    'head_nums': 4, #heads number of multihead attention
    'encoder_nums': 12, #number of transformer encoders
    'classes': 14, #number of classes
    'dropout_probs': .0 #dropout probability
}