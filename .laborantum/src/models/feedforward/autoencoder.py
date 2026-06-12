import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()

        ## YOUR CODE HERE
        encoder_layers = []
        for index in range(len(channels) - 1):
            encoder_layers.append(
                torch.nn.Linear(channels[index], channels[index + 1])
            )
            if index < len(channels) - 2:
                encoder_layers.append(activation())

        decoder_channels = list(reversed(channels))
        decoder_layers = []
        for index in range(len(decoder_channels) - 1):
            decoder_layers.append(
                torch.nn.Linear(decoder_channels[index], decoder_channels[index + 1])
            )
            if index < len(decoder_channels) - 2:
                decoder_layers.append(activation())

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Identity()
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Identity()

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal

        ## YOUR CODE HERE
        res = res.reshape(res.shape[0], -1)
        res = self.encoder(res)
        res = self.decoder(res)

        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            batch['signals'] = {}

        batch['signals']['reconstruction'] = self.__forward_kernel(
            batch['data']['image']
        )

        return batch
