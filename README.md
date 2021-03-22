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

### Usage

#### Prerequisite Python packages:
* `torch`
* `torchvision`
* `tqdm`
* `tensorboard`
* `pillow`
* `numpy`

#### Brief repository overview

Project structure:
* `configs/` -
* `data/` -
* `datasets/` - created by default to store downloaded and extracted datasets (can be configured to be stored somewhere else)
* `experiments/` - created by default to store checkpoints and `tensorboard` logs of different training experiments (can be configured to be stored somewhere else)
* `models/` - contains all models that we tried to train using Dynamic Convolutions, models derive from `BaseModel` class which allows to construct models with a custom convolutional layers (either `nn.Conv2d` or out `DynamicConvolution`), models' sub modules derive from `TempModule` class, which allows pass a `temperature` argument to `forward()` method 
    * `models/deeplab_details` - contains code primarily originated from [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception) project (their backbones setups and model architecture details). We enhance those details (not backbones) it with possibility to apply Dynamic Convolutions and a width multiplier on convolutional layers
* `notebooks/` -
* `utils/` -
* `dynamic_convolutions.py` -
* `inspect_attention.py` -
* `train.py` - entry point for training of models, using a configuration file (or other options)

#### Reproduce training
The most basic way is to enter the root of the project and execute the following in a console:
```
python train.py --config_path "configs/<name>.yaml"
```
where `<name>.yaml` refers to a filename among those lying in `configs/` folder. There you can find different setups of training, namely:
* 🔴 TODO: other configs
* `mobilenetv3.yaml` - baseline of MobileNetV3 🔴 TODO: elaborate on this
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
