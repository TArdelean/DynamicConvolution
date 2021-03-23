from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def MNIST_dataset(stage="train", download=True, root='datasets/'):
    if stage == "train":
        return datasets.MNIST(root, train=True, download=download,
                              transform=transform)
    else:
        return datasets.MNIST(root, train=False, download=download,
                              transform=transform)
