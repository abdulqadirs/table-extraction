from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from torch import optim
from config import Config
from utils.read_config import reading_config
from data import data_loaders, test_data_loader
from models.split.split import SplitModel
from deep_splitting import train, testing
from optimizer import adam_optimizer
# from inference import inference
from postprocessing.ocr import img_to_hocr
from postprocessing.table_extraction import extract_csv
from utils.parse_arguments import parse_training_arguments



def main():

    args, _ = parse_training_arguments()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    if args.checkpoints is not None:
        checkpoints_file = Path(args.checkpoints)
    
    if args.training:
        config_file = Path('../config.ini')
        reading_config(config_file)
        training_loader, validation_loader = data_loaders(images_dir, labels_dir)
        split_model = SplitModel(input_channels=1)
        net = split_model.to(Config.get('device'))
        learning_rate = Config.get('learning_rate')
        weight_decay = Config.get('weight_decay')
        optimizer = adam_optimizer(net, learning_rate, weight_decay)
        if args.checkpoints is not None:
            checkpoint_file = Path(args.checkpoints)
            checkpoint = torch.load(checkpoint_file, map_location=Config.get('device'))
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train(net, training_loader, validation_loader, optimizer, epochs, start_epoch, validate_every)
        else:
            start_epoch = 1
            epochs = 2
            validate_every = 1
            train(net, training_loader, validation_loader, optimizer, epochs, start_epoch, validate_every, output_dir)
    
    elif args.testing:
        config_file = Path('../config.ini')
        reading_config(config_file)
        testing_loader = test_data_loader(images_dir, labels_dir)
        split_model = SplitModel(input_channels=1)
        net = split_model.to(Config.get('device'))
        checkpoint_file = Path(args.checkpoints)
        checkpoint = torch.load(checkpoint_file, map_location=Config.get('device'))
        testing(net, testing_loader)






if __name__ == "__main__":
    main()