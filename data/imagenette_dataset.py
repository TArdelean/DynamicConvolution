import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def Imagenette_dataset(stage="train"):
    if stage == "train":
        return ImagenetteDataset('datasets/', split="train", download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomAffine(15, None, (0.9, 1.1)),
                                       transforms.RandomResizedCrop(224, scale=(0.25, 1.0)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
                                       transforms.GaussianBlur(5, (0.1, 0.5)),
                                       transforms.ToTensor(),
                                       normalize
                                   ]))
    else:
        return ImagenetteDataset('datasets/', split="val", download=True,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize
                                   ]))


class ImagenetteDataset(ImageFolder):
    """
        Dataset for Imagenette: a subset of 10 easily classified classes from Imagenet
    """
    base_folder = 'imagenette2-320'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'imagenette2-320.tgz'
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'

    def __init__(self, root, split='train', download=False, **kwargs):
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url, self.data_root, filename=self.filename,
            remove_finished=True)
