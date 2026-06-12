import torch
import torchvision.datasets


class MNISTSimpleDataset:
    def __init__(self, train=True):
        dataset = torchvision.datasets.MNIST(
            root=".",
            train=train,
            download=True,
        )

        self.X = dataset.data
        self.y = dataset.targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index].to(torch.float32)
        image = image / 127.5 - 1.0

        label = self.y[index].to(torch.long)

        return {
            "image": image,
            "label": label,
        }
