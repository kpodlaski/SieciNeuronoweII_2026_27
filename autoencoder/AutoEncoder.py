from autoencoder.Decoder import Decoder
from autoencoder.Encoder import Encoder
import torch.nn as nn

class AutoencoderNet(nn.Module):
    def __init__(self, encoder_dim):
        super(AutoencoderNet, self).__init__()
        self.code_dim = encoder_dim
        self.encoder = Encoder(encoded_space_dim=encoder_dim,fc2_input_dim=128)
        self.decoder = Decoder(encoded_space_dim=encoder_dim,fc2_input_dim=128)
        self.params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

    def parameters(self):
        return [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)