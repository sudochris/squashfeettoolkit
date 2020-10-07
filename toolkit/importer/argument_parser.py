import argparse


class ArgumentOption:
    def __init__(self, argument_name, help_string, default_value, action=None) -> None:
        super().__init__()
        self.argument_name = argument_name
        self.specification = {}

        self.specification.update({"dest": argument_name})
        self.specification.update({"default": default_value})
        self.specification.update({"help": help_string})
        self.specification.update({"dest": argument_name})
        if action is not None:
            self.specification.update({"action": action})

class StringOption(ArgumentOption):
    def __init__(self, argument_name, help_string="", default_value="") -> None:
        super().__init__(argument_name, help_string, default_value, None)


class BooleanOption(ArgumentOption):
    def __init__(self, argument_name, help_string="", default_value=False) -> None:
        super().__init__(argument_name, help_string, default_value, "store_true")

class ApplicationSettings:
    def __init__(self, parsed_args) -> None:
        super().__init__()
        self.parsed_args = parsed_args

    def description_file(self):
        return self.parsed_args.description

    def debug(self):
        return self.parsed_args.debug

    def render(self):
        return self.parsed_args.render

    def output_folder(self):
        return self.parsed_args.output


def application_settings_from_args(argument_options):
    parser = argparse.ArgumentParser(description='')

    for argument_option in argument_options:
        parser.add_argument(f"--{argument_option.argument_name}", **argument_option.specification)

    return ApplicationSettings(parser.parse_args())