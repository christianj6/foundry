from ..run import run
import argparse


class ManagementUtility:
    def __init__(self, argv: list = None) -> None:
        self.argv = argv
        self.parser = self.setup_parser()

    @classmethod
    def setup_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-k",
            "--keywords",
            type=str,
            help="Absolute filepath to .csv file containing keywords.",
        )
        parser.add_argument(
            "-c",
            "--corpus",
            type=str,
            help="Absolute path to .csv file containing corpus of documents to search.",
        )
        parser.add_argument(
            "-l",
            "--language",
            type=str,
            help="Full lowercase name of the text language. Used for removing stopwords.",
        )
        parser.add_argument(
            "-b",
            "--bound",
            type=float,
            help="Lower bound for matching cutoff. A float value between 0.0 and 1.0.",
        )
        parser.add_argument(
            "-t",
            "--train",
            action="store_true",
            help="Switch for running training task.",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            help="Absolute path to model binary file.",
        )
        parser.add_argument(
            "-d", "--data", type=str, help="Data to use for training."
        )
        parser.add_argument(
            "-e",
            "--evaluate",
            action="store_true",
            help="Run evaluation task against provided model.",
        )

        return parser

    def execute(self) -> None:
        args = self.parser.parse_args()
        run(**vars(args))


def execute_from_command_line(argv: list = None) -> None:
    utility = ManagementUtility(argv)
    utility.execute()
