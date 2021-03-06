from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def MNIST_dataset(stage="train"):
    if stage == "train":
        return datasets.MNIST('datasets/', train=True, download=True,
                              transform=transform)
    else:
        return datasets.MNIST('datasets/', train=False,
                              transform=transform)
