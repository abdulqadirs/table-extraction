from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

def parse_arguments():
    """
    Parses the arguments passed through the terminal.
    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--input_images', help='Input table images directory', required=True)
    parser.add_argument('-x', '--xml_labels', help='Path of folder containing xml labels', required=True)
    parser.add_argument('-p', '--processed_images', help='Path of output processed images', required=True)
    parser.add_argument('-j', '--json_labels', help='Path of output processed labels', required=True)

    return parser.parse_known_args()