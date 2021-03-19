
# ALL ACKNOWLEDGMENT GOES TO THE PAPER & REPOSITORY AUTHORS
# https://github.com/jfzhang95/pytorch-deeplab-xception

from torchvision import transforms
from torchvision import datasets

import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

def SB_dataset(stage="train", download=True, root='datasets/SBD'):
    if stage == "train":
        return datasets.SBDataset(root, image_set='train_noval',
                                download=download, mode='segmentation',
                                transforms=CustomCompose([
                                    CustomRandomHorizontalFlip(),
                                    # NOTE: original repo has args parameter  
                                    # CustomRandomScaleCrop(base_size=args.base_size, crop_size=args.crop_size),
                                    CustomRandomScaleCrop(base_size=200, crop_size=200),
                                    CustomRandomGaussianBlur(),
                                    CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    CustomToTensor(),
                                ]))
    else:
        return datasets.SBDataset(root, image_set='val', 
                                download=download, mode='segmentation',
                                transforms=CustomCompose([
                                    CustomFixScaleCrop(crop_size=200),
                                    CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    CustomToTensor(),
                                ]))

class CustomRandomGaussianBlur(object):
    def __call__(self, img, mask):
        #img, mask = sample
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img, mask


class CustomRandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img, mask):
        #img, mask = sample
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask

class CustomFixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        #img, mask = sample
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask

class CustomToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #img, mask = sample
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.LongTensor(mask)
        return img, mask


class CustomRandomHorizontalFlip(object):
    def __call__(self, img, mask):
        #img, mask = sample
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask

class CustomNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img, mask):
        #img, mask = sample
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img, mask

class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    sbd_train = SB_dataset(stage='train', download=False)
    print('Created dataset')
    dataloader = DataLoader(sbd_train, batch_size=2, shuffle=True, num_workers=0)
    print('Created loader')
    for ii, sample in enumerate(dataloader):
        img, gt = sample
        for jj in range(img.size()[0]):
            plt.figure()
            plt.subplot(211)
            plt.imshow(img[jj].numpy().transpose((1, 2, 0)))
            plt.subplot(212)
            plt.imshow(gt[jj].numpy())
        break
    plt.show(block=True)