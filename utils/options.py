from utils.parsable_options import ParsableOptions


class Options(ParsableOptions):

    # noinspection PyAttributeOutsideInit
    def initialize(self):
        self.nof_kernels = 4
        self.model_class = ""
        self.dataset_class = ""  # Can also be a function which returns a dataset instance
        self.batch_size = 16
        self.num_workers = 2
        self.lr = 0.001
        self.device = "cpu"


opt = Options(config_file_arg="config_path")
