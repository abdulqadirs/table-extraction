from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from torch import optim
from config import Config
from utils.read_config import reading_config
from data import data_loaders
from models.split.split import SplitModel
from deep_splitting import train
from optimizer import adam_optimizer
from inference import inference
from postprocessing.ocr import img_to_hocr
from postprocessing.table_extraction import extract_csv


def main():

    #config_file = Path('../config.ini')
    #reading_config(config_file)

    # root_dir = '../../dataset/preprocessed-data-0'
    # images_dir = '../../dataset/preprocessed-data-0/input-jpg'
    # xml_dir = '../../dataset/preprocessed-data-0/json-labels'

    # training_loader, validation_loader, testing_loader = data_loaders(root_dir, images_dir, xml_dir)
    # split_model = SplitModel(input_channels=1)
    # net = split_model.to(Config.get('device'))
    # learning_rate = Config.get('learning_rate')
    # weight_decay = Config.get('weight_decay')
    # optimizer = adam_optimizer(net, learning_rate, weight_decay)
    # #checkpoint_file = '/gdrive/My Drive/deep-splitting-merging/model/checkpoint.deep_column_splitting.pth.tar'
    # #checkpoint = torch.load(checkpoint_file, map_location=Config.get('device'))
    # #start_epoch = checkpoint['epoch'] + 1
    # #net.load_state_dict(checkpoint['net'])
    # #optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = 1
    # epochs = 50
    # validate_every = 5

    # train(net, training_loader, validation_loader, optimizer, epochs, start_epoch, validate_every)

    orig_path = Path('../../dataset/preprocessed-data-0/input-jpg/1.jpg')
    resized_path = Path('../../dataset/test/img.jpg')
    hocr_path = Path('../../dataset/test/output')
    seg_path = Path('../../dataset/test/segmented.jpg')
    df = inference(orig_path, resized_path, hocr_path, seg_path)
    print(df)




if __name__ == "__main__":
    main()