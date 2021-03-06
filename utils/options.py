import os

from utils.parsable_options import ParsableOptions


class Options(ParsableOptions):

    # noinspection PyAttributeOutsideInit
    def initialize(self):
        self.nof_kernels = 4
        self.experiments = "experiments"
        self.model_class = ""
        self.dataset_class = ""  # Can also be a function which returns a dataset instance
        self.experiment_name = "attempt"
        self.checkpoint_path = "experiments/attempt/attempt_4.pth"
        self.save_freq = 1
        self.batch_size = 16
        self.num_workers = 2
        self.lr = 0.001
        self.device = "cpu"

    # noinspection PyAttributeOutsideInit
    def proc(self):
        self.checkpoints_dir = os.path.join(self.experiments, self.experiment_name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.print_to_file()

    def print_to_file(self, **kwargs):
        super(Options, self).print_to_file(
            os.path.join(self.checkpoints_dir, "config.yaml"))


opt = Options(config_file_arg="config_path")
