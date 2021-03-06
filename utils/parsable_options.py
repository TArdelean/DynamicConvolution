import numbers
import argparse


class ParsableOptions:
    def __init__(self, suppress_parse=False, config_file_arg=None):
        """
        Base class for parsable options
        :param suppress_parse: Whether to parse CLI arguments. Useful when passing external options
        :param config_file_arg: Allows reading existing configuration from file.
        """
        if config_file_arg is not None:
            self.__setattr__(config_file_arg, "")
        self._config_file_arg = config_file_arg
        self.initialize()
        if not suppress_parse:
            self.parse()
            self.proc()

    def initialize(self):
        """ Method were all fields should be initialized. Variables starting with _ will not be parsed"""
        pass

    def proc(self):
        """ Post processing, after the options have been parsed """
        pass

    @staticmethod
    def good_instance(val):
        return isinstance(val, str) or (isinstance(val, numbers.Number) and not isinstance(val, bool))

    def parse(self):
        parser = argparse.ArgumentParser()
        for name, val in vars(self).items():
            if name.startswith("_"):
                continue
            like = type(val) if ParsableOptions.good_instance(val) else eval
            parser.add_argument(f'--{name}', type=like, default=argparse.SUPPRESS, help="It is obvious")

        args = parser.parse_args()
        if self._config_file_arg is not None:
            config = getattr(args, self._config_file_arg, "")
            if config:
                self.load_from_file(config)
        for name, val in vars(self).items():
            if name.startswith("_"):
                continue
            if name in args:
                attr = getattr(args, name)
                self.__setattr__(name, attr)

    def print_to_file(self, file_path, include_hidden=False):
        """ Prints options to a config file in a human readable format """
        with open(file_path, "w") as f:
            for name, val in vars(self).items():
                if name.startswith("_") and not include_hidden:
                    continue
                val_format = str(val) if not isinstance(val, str) else f"'{val}'"
                f.write(f"{name}: {val_format}\n")

    def load_from_file(self, file_path, include_hidden=False):
        """ Load options from config file"""
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:  # First and last lines are {}
                tokens = line.split(":")  # Getting rid of \t and \n
                identifier = tokens[0].strip()
                value = tokens[1].strip()
                if not hasattr(self, identifier):
                    print(f"Warning: redundant option {identifier}")
                    continue
                if identifier.startswith("_") and not include_hidden:
                    continue
                if value.startswith("'"):
                    parsed_value = value[1:-1]
                else:
                    parsed_value = eval(value)
                self.__setattr__(identifier, parsed_value)