from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

def parse_arguments():
    """
    Parses the arguments passed through the terminal.
    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--input_path', help='Path of the input pdf or scanned images.', required=True)
    parser.add_argument('-o', '--output_path', help='Path of output csv file.', required=True)
    parser.add_argument('-f', '--finetuned_ocr', help='Directory where fineturned tesseract ocr is stored.', required=True)
    
    #pdf or scanned image.
    mode_parser = parser.add_mutually_exclusive_group(required=True)
    mode_parser.add_argument('-p', '--pdf', help='Input is pdf.', action='store_true')
    mode_parser.add_argument('-s', '--scanned-pdf', help='Input is image.', action='store_true')

    return parser.parse_known_args()