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

* [`configs/`](configs/) - configuration files to be used with `train.py` to train different combinations of models / datasets / optimizers / schedulers and their parameters
* [`data/`](data/) - datasets downloading and setting up:
   * `imagenette_dataset.py` - Imagenette dataset ([download](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)), which is a small ImageNet subset of 10 classes. 
   * `tinyimagenet_dataset.py` - Tiny ImageNet dataset ([download](http://cs231n.stanford.edu/tiny-imagenet-200.zip)),  which is a small ImageNet subset of 200 classes with 600 images per class. 
   * `pascalvoc2012_dataset.py` - Pascal VOC 2012 segmentation dataset ([official page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)), used to train and evaluate DeepLabV3+ 
   * `sb_dataset.py` - Semantic Boundaries segmentation dataset ([official page](http://home.bharathh.info/pubs/codes/SBD/download.html)), used to train DeepLabV3+
   * `mnist_dataset.py` - MNIST dataset, provided to test correctness of installation with a trivial classification CNN
* [`datasets/`](datasets/) -default storage of downloaded and extracted datasets (can be configured to be stored somewhere else)
* [`experiments/`](experiments/) - default storage of checkpoints and `tensorboard` logs of experiments (can be configured to be stored somewhere else)
* [`models/`](models/) - different models' source code on which we apply Dynamic Convolutions
   * `common.py` - base classes that allow integration of Dynamic Convolutions into existing models. Models derive from `BaseModel` class which allows to construct a model with a custom convolutional layer class (either `nn.Conv2d` or our `DynamicConvolution`), submodules inside models derive from `TempModule` class, which allows pass a `temperature` argument to `forward()` method 
   * `deeplab.py` and `deeplab_details/` folder - [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
   * `mobilenetv3.py` - MobileNetV3
   * `mobilenetv2.py` - MobileNetV2
   * `resnet.py` - general ResNet, ResNet10 and ResNet18
   * `simple_conv.py` - a trivial classification CNN provided to test correctness of installation 
* [`tensorboard_events/`](tensorboard_events/) - a collection of training logs: loss and scores progression over epochs, made to be parsed/visualized in TensorBoard library  
* [`notebooks/`](notebooks/) - check out section [below](#notebooks)
* [`utils/`](utils/) - framework auxiliary code for parsing options from configuration files and command line, loading and storing of checkpoints, custom metrics, losses
* 👉[`dynamic_convolutions.py`](dynamic_convolutions.py) - implementation of `DynamicConvolution` class, i.e. a drop-in replacement for `nn.Conv2d` with learnable per-sample attention
* `inspect_attention.py` - methods to extract and analyze Dynamic Convolution state in trained models 
* 👉 [`train.py`](train.py) - entry point for training of models, using a configuration file (or other options, see **Reproduce training** section)


### Requirements 
A GPU is recommended to perform the experiments. 
The code is set up to be effortlessly run using [Google Colab](colab.research.google.com). 
Main prerequisites are:

- [`pytorch`](http://pytorch.org/), [`torchvision`](https://github.com/pytorch/vision)
- `numpy`, `tensorboard`, `pillow`, `tqdm`

For convenience, a ready to use conda [environment](environment.yml) is provided. 
To create a new python environment with all the required packages, you can run:
```shell
conda env create -f environment.yml
conda activate dyconv
```

### Training setup 
The training process can be started with the provided script:
```shell
python train.py --config_path "configs/<name>.yaml"
```
where `<name>.yaml` refers to a configuration file ([`configs/`](configs)) facilitating experiment reproducibility. Among the available training setups there are: 
* `MobileNetV2_[type]_[width-multiplier]_[temperature]` - various setups of MobileNetV2 architecture
* `MobileNetV3_[type]_[width-multiplier]_[dataset]` - various setups of MobileNetV3 architecture
* `deeplabv3plus.yaml`, `dy_deeplabv3plus.yaml` - DeepLabV3+ with MobileNetV2 backbone trained on Pascal VOC 2012 dataset (and Dynamic Convolution variant)
* `deeplabv3plus_0.5.yaml`, `dy_deeplabv3plus_0.5.yaml` - baseline of DeepLabV3+ with MobileNetV2 backbone and x0.5 width convolutions, trained on a combination of Pascal VOC 2012 and Semantic Boundaries datasets (and Dynamic Convolution variant)
* `Resnet_[type]_[width-multiplier]` - various setups of ResNet architecture
* `config.yaml` - toy example of classifying MNIST digits with a trivial CNN, provided to test correctness of installation

Besides providing the config file, you can also pass options as command line arguments, which will override those used in the config. For the full list of available (and default) options refer to [`utils/options.py`](utils/options.py) file. 
```shell
python train.py --config_path "configs/<name>.yaml" --device "cpu" --batch_size 100
```

TensorBoard facilitates tracking the training process and comparing experiments:
```shell
tensorboard --logdir experiments
```

### Notebooks

The following Google Colab compatible Jupyter notebooks are available:
- [`notebooks/DyConv_training.ipynb`](notebooks/DyConv_training.ipynb) - Self-explanatory training procedure
- [`notebooks/DyConv_profiling.ipynb`](notebooks/DyConv_profiling.ipynb) - Profile number of parameters, FLOPS and inference time
- [`notebooks/DyConv_inspect.ipynb`](notebooks/DyConv_inspect.ipynb) - Inspect the distribution of attentions computed by Dynamic Convolutions
- [`notebooks/DyConv_inspect_segmentation.ipynb`](notebooks/DyConv_inspect_segmentation.ipynb) - Inspect learning progress and statistics of segmentation models training (DeepLabV3+)

### Experimental results

#### Classification

- Dataset: Tiny ImageNet

  | Network                 |  Number of parameters |  Accuracy  |         Config file       |   
  | :---                    |         :---:         |    :---:   |            :---:          |
  | ResNet-10               |         5.00M         |    56.86   |  `Resnet_Vanilla_1.yaml`  |
  | DY-ResNet-10            |        19.80M         | **58.17**  |     `Resnet_DY_1.yaml`    |
  | ResNet-10 x 0.5         |         1.27M         |    52.78   |  `Resnet_Vanilla_05.yaml` |
  | DY-ResNet-10 x 0.5      |         4.97M         |    53.92   |  `Resnet_DY_05.yaml`      |
  | DY-ResNet-10 x 0.5 (**) |         4.97M         |    55.38   |`Resnet_DY_Leaky_05_smooth.yaml`|
  | ResNet-10 x 0.25        |         0.33M         |    46.17   | `Resnet_Vanilla_025.yaml` |
  | DY-ResNet-10 x 0.25     |         1.25M         |    48.05   |    `Resnet_DY_025.yaml`   |
  | MobileNetV2 x 0.35      |         0.65M         |    34.05   |  `MobileNetV2_Vanilla_035.yaml`|
  | DY-MobileNetV2 x 0.35   |         2.09M         |    31.91   |  `MobileNetV2_DY_035.yaml`|
  | MobileNetV3 x 0.5       |         0.87M         |    53.36   |  `MobileNetV3_Vanilla_05_Tiny.yaml`|
  | DY-MobileNetV3 x 0.5    |         2.52M         |    50.95   |  `MobileNetV3_DY_05_Tiny.yaml` |
  | MobileNetV3             |         1.91M         |    55.53   |  `MobileNetV3_Vanilla_1_Tiny.yaml` |
  | DY-MobileNetV3          |         6.04M         |    53.59   |  `MobileNetV3_DY_1_Tiny.yaml` |

- Dataset: Imagenette

  | Network                 |  Number of parameters |  Accuracy  |         Config file       |   
  | :---                    |         :---:         |    :---:   |            :---:          |
  | MobileNetV3 x 0.25      |         0.27M         |    85.07   |  `MobileNetV3_Vanilla_025_Imagenette.yaml` |
  | DY-MobileNetV3 x 0.25   |         0.99M         |    85.24   |  `MobileNetV3_DY_025_Imagenette.yaml` |
  | MobileNetV3 x 0.5       |         0.63M         |    87.31   |  `MobileNetV3_Vanilla_05_Imagenette.yaml` |
  | DY-MobileNetV3 x 0.5    |         2.28M         |    87.89   |  `MobileNetV3_DY_05_Imagenette.yaml` |
  | MobileNetV3             |         1.67M         |    88.40   |  `MobileNetV3_Vanilla_1_Imagenette.yaml` |
  | DY-MobileNetV3          |         5.80M         |  **89.32** |  `MobileNetV3_DY_1_Imagenette.yaml` |

#### Segmentation

- Dataset: Pascal VOC 2012

  | Network             | Number of parameters  |  mIoU   | Config file | 
  | :---                |         :---:         |   :---: |    :---:    |
  | DeepLabV3+          |         5.81M         |  65.09  | `deeplabv3plus.yaml`|
  | DY-DeepLabV3+       |         18.4M         |  **65.45**  | `dy_deeplabv3plus.yaml`|


- Dataset: Pascal VOC 2012 + SBD (only used for training)
  
  | Network             | Number of parameters  |  mIoU  | Config file | 
  | :---                |         :---:         |  :---: |    :---:    |
  | DeepLabV3+ x 0.5    |         3.40M         |  70.23 | `deeplabv3plus_0.5.yaml` |
  | DY-DeepLabV3+ x 0.5 |         8.44M         |  **71.09** | `dy_deeplabv3plus_0.5.yaml` |
