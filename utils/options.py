import os

from utils.parsable_options import ParsableOptions


class Options(ParsableOptions):

    # noinspection PyAttributeOutsideInit
    def initialize(self):
        self.use_dynamic = False
        self.nof_kernels = 4  # Parameter is ignored if not using dynamic
        self.reduce = 4  # Dimension reduction in hidden layer for attention in dynamic convolutions
        self.temperature = (30, 1, 10)  # Temperature parameters: (initial_value, final_value, final_epoch)
        self.experiments = "experiments"
        self.model_class = ""
        self.model_extra_args = ()  # Model specific arguments (e.g. width multiplier)
        self.dataset_class = ""  # Can also be a function which returns a dataset instance
        self.experiment_name = "attempt"
        self.checkpoint_path = None  # e.g. "experiments/attempt/attempt_4.pth"
        self.max_epoch = 10
        self.save_freq = 1
        self.batch_size = 16
        self.num_workers = 2
        self.optimizer = "SGD"
        self.optimizer_args = (0.001, 0.9)  # e.g. (lr, momentum)
        self.scheduler = "StepLR"
        self.scheduler_args = (30, 0.1)  # e.g. (step_size, gamma) for StepLR
        self.criterion = "SmoothNLLLoss"
        self.criterion_args = (0.1,)
        self.device = "cpu"
        self.is_classification = True # otherwise segmentation
        self.n_classes = None         # NOTE(alexey-larionov): added for segmentation model and dataset
        self.config_path = ""
        self.download_dataset = True
        self.dataset_root = './datasets'
        self.dataset_extra_args = ()

    # noinspection PyAttributeOutsideInit
    def proc(self):
        super().proc()
        self.checkpoints_dir = os.path.join(self.experiments, self.experiment_name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.print_to_file()

    def print_to_file(self, **kwargs):
        super(Options, self).print_to_file(
            os.path.join(self.checkpoints_dir, "config.yaml"))
