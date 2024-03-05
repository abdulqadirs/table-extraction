import configparser
import logging
import torch

from config import Config

logger = logging.getLogger('segmentation')

def reading_config(file_path):
    """
    Reads the config settings from config file and makes them accessible to the project using config.py module.
    Args:
        file_path (Path): The path of config file.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    # TODO (aq): Raise Error if file doesn't exist. 
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        logger.info('Reading the config file from: %s' % file_path)
    except FileNotFoundError:
        logger.exception("Config file doesn't exist.")

    #GPUs
    Config.set("disable_cuda", config.getboolean("GPU", "disable_cuda", fallback=True))
    if not Config.get("disable_cuda") and torch.cuda.is_available():
        Config.set("device", "cuda")
        logger.info('GPU is available')
        print('GPU available')
    else:
        Config.set("device", "cpu")
        logger.info('Only CPU is available')
        print('Only cpu is available')

    #paths
    # Config.set("images_dir", config.get("paths", "images_dir", fallback="Flickr8k_images"))
    # Config.set("captions_dir", config.get("paths", "captions_dir", fallback="Flickr8k_text/Flickr8k.token.txt"))
    # Config.set("output_dir", config.get("paths", "outdir", fallback="output"))
    # Config.set("checkpoint_file", config.get("paths", "checkpoint_file", fallback="checkpoint.ImageCaptioning.pth.tar"))

    #split_model
    Config.set("image_channels", config.getint("split_model", "image_channels", fallback=1))
    Config.set("max_width", config.getint("split_model", "max_width", fallback=1800))
    Config.set("max_height", config.getint("split_model", "max_height", fallback=1800))
    Config.set("resize_scale", config.getfloat("split_model", "resize_scale", fallback=0.9))
    Config.set('min_width', config.getint('split_model', 'min_width', fallback=1500))
    Config.set("num_of_modules", config.getint("split_model", "num_of_modules", fallback=5))

    #Training
    Config.set("training_batch_size", config.getint("training", "batch_size", fallback=1))
    Config.set("epochs", config.getint("training", "epochs", fallback=8))
    Config.set("learning_rate", config.getfloat("training", "learning_rate", fallback=0.00075))
    Config.set("weight_decay", config.getfloat("training", "weight_decay", fallback=0.001))
    Config.set("drop_out", config.getfloat("training", "drop_out", fallback=0.4))

    #validation
    Config.set("validation_batch_size", config.getint("validation", "batch_size", fallback=1))
    Config.set("validate_every", config.getint("validation", "validate_every", fallback=1))

    #testing
    Config.set("testing_batch_size", config.getint("testing", "batch_size", fallback=1))

    #logging
    Config.set("logfile", config.get("logging", "logfile", fallback="output.log"))