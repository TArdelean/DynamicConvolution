# Dynamic Convolution

Unofficial implementation of Dynamic Convolutions. Approach from paper 
[Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/pdf/1912.03458.pdf).

Source code of the Project for the Machine Learning course at Skoltech 2021.


### Team members

- Timotei Ardelean
- Andreea Dogaru
- Alexey Larionov
- Oleg Maslov
- Saian Protasov 

### Brief repository overview

* `configs/` - configuration files that allow to launch script `train.py` to train different combinations of models/datasets/optimizers/schedulers and their parameters
* `data/` - classes for downloading and setting up different datasets:
   * `imagenette_dataset.py` - Imagenette dataset, which is a small ImageNet subset of 10 classes. [Download link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)
   * `tinyimagenet_dataset.py` - Tiny ImageNet dataset,  which is a small ImageNet subset of 200 classes with 600 images per class. [Download link](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
   * `pascalvoc2012_dataset.py` - Pascal VOC 2012 segmentation dataset, used to train and evaluate DeepLabV3+ model. [Official page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
   * `sb_dataset.py` - Semantic Boundaries dataset for segmentation, DeepLabV3+ originally is trained on samples from here together with samples from Pascal VOC 2012. [Official page](http://home.bharathh.info/pubs/codes/SBD/download.html)
   * `mnist_dataset.py` - MNIST dataset, provided to test correctness of installation with a trivial classification CNN
* `datasets/` - may be created by default to store downloaded and extracted datasets (can be configured to be stored somewhere else)
* `experiments/` - may be created by default to store checkpoints and `tensorboard` logs of different training experiments (can be configured to be stored somewhere else)
* `models/` - contains all models source code that we tried to train using Dynamic Convolutions
   * `common.py` - base classes that allow integration of Dynamic Convolutions into existing models. Models derive from `BaseModel` class which allows to construct a model with a custom convolutional layer class (either `nn.Conv2d` or out `DynamicConvolution`), submodules inside models derive from `TempModule` class, which allows pass a `temperature` argument to `forward()` method 
   * `deeplab.py` and `deeplab_details/` folder - [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
   * `mobilenetv3.py` - MobileNetV3
   * `mobilenetv2.py` - MobileNetV2
   * `resnet.py` - general ResNet, ResNet10 and ResNet18
   * `simple_conv.py` - a trivial classification CNN provided to test correctness of installation 
* `notebooks/` - 🔴 TODO
* `utils/` - framework auxiliary code for parsing options from configuration files and command line, loading and storing of checkpoints, custom metrics, losses
* 👉`dynamic_convolutions.py` - implementation of `DynamicConvolution` class, i.e. a drop-in replacement for `nn.Conv2d` with learnable per-sample attention
* `inspect_attention.py` - methods to extract and analyze Dynamic Convolution state in trained models 
* 👉 `train.py` - entry point for training of models, using a configuration file (or other options, see **Usage** section)

### Usage

#### Prerequisite Python packages:
* `torch`, `torchvision`, `tqdm`, `tensorboard`, `pillow`, `numpy`

#### Reproduce training
The most basic way is to enter the root of the project and execute the following in a console:
```
python train.py --config_path "configs/<name>.yaml"
```
where `<name>.yaml` refers to a filename among those lying in `configs/` folder. There you can find different setups of training, namely:
* 🔴 TODO: other configs
* `mobilenetv3.yaml` - baseline of MobileNetV3 🔴 TODO: elaborate on this?
* `mobilenetv2.yaml` - baseline of MobileNetV2 🔴 TODO: elaborate on this?
* `deeplabv3plus.yaml` - baseline of DeepLabV3+ with MobileNetV2 backbone trained on Pascal VOC 2012 dataset
* `dy_deeplabv3plus.yaml` - Dynamic Convolution version of DeepLabV3+ with MobileNetV2 backbone trained on Pascal VOC 2012 dataset
* `deeplabv3plus_0.5.yaml` - baseline of DeepLabV3+ with MobileNetV2 backbone and x0.5 width convolutions, trained on a combination of Pascal VOC 2012 and Semantic Boundaries datasets
* `dy_deeplabv3plus_0.5.yaml` - Dynamic Convolution version of DeepLabV3+ with MobileNetV2 backbone and x0.5 width convolutions, trained on a combination of Pascal VOC 2012 and Semantic Boundaries datasets
* `config.yaml` - toy example of classifying MNIST digits with a trivial CNN, provided to test correctness of installation

Besides providing the config file, you can also pass options as command line arguments, which will override those used in the config. For example
```
python train.py --config_path "configs/<name>.yaml" --device "cpu" --batch_size 100
```
For the full list of available (and default) options refer to `utils/options.py` file

Alternatively you can look over `notebooks/` directory where we have Jupyter notebooks with ability to do either afomementioned traning from a config on Google Colab, or adjust some training parameters for custom training. 

#### Checkpoints 🔴 TODO: maybe delete
We publish some of the checkpoints for trained models

#### Some results 🔴 TODO: fill in (required)
