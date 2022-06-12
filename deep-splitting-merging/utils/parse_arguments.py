from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

def parse_preprocessing_arguments():
    """
    Parses the data preprocessing arguments passed through the terminal.
    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--input_images', help='Input table images directory', required=True)
    parser.add_argument('-x', '--xml_labels', help='Path of folder containing xml labels', required=True)
    parser.add_argument('-p', '--processed_images', help='Path of output processed images', required=True)
    parser.add_argument('-j', '--json_labels', help='Path of output processed labels', required=True)

    return parser.parse_known_args()


def parse_training_arguments():
    """
    Parses the training arguments passed through the terminal.
    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--images_dir', help='Path of folder containing processed input images', required=True)
    parser.add_argument('-l', '--labels_dir', help='Path of folder containing json labels', required=True)
    parser.add_argument('-o', '--output_dir', help='Path of output directory to save intermediate results or checkpoints', 
                            required=True)                      
    parser.add_argument('-c', '--checkpoints', help='Full path with filename of checkpoint file for testing')

    #train or test the model
    mode_parser = parser.add_mutually_exclusive_group(required=True)
    mode_parser.add_argument('-t', '--training', help='Train the model.', action='store_true')
    mode_parser.add_argument('-e', '--testing', help='Test the model.', 
                                action='store_true')

    args, _ = parser.parse_known_args()
    if args.testing and args.checkpoints is None:
        parser.error('Provide the path of checkpoints file for testing')

    return parser.parse_known_args()


def parse_inference_arguments():
    """
    Parses the  inference arguments passed through the terminal.
    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--image_path', help='Path of the input table image.', required=True)
    parser.add_argument('-c', '--checkpoints', help='Full path of checkpoint file for testing', required=True)
    parser.add_argument('-o', '--output_dir', help='Path of output directory', required=True)                      

    return parser.parse_known_args()