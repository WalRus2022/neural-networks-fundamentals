import torch


class SimpleFCNN(torch.nn.Module):
    def __init__(
            self,
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU):
        super().__init__()

        if channels is None:
            channels = [784, 128, 64]

        layers = []

        for in_features, out_features in zip(channels[:-1], channels[1:]):
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(activation())

        layers.append(torch.nn.Linear(channels[-1], n_classes))

        self.layers = torch.nn.Sequential(*layers)

    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.layers(signal)

        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        batch['signals'] = {'output': signal}

        self.postprocessing(batch)

        return batch['signals']['output']

    def postprocessing(self, batch):
        signal = batch['signals']['output']
        signal = torch.argmax(signal, dim=1)

        batch['postprocessed'] = {'class': signal}