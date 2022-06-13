import numpy as np
import cv2
import torch 
from PIL import Image
from config import Config
from pathlib import Path
import os
from utils.read_config import reading_config
from models.split.split import SplitModel
from preprocess import preprocess_img
from postprocessing.ocr import img_to_hocr
from postprocessing.table_extraction import extract_csv
from utils.parse_arguments import parse_inference_arguments
import torchvision.transforms as transforms


def resize_img(img):
    img = np.array(img)
    h, w = img.shape
    min_width = 400
    scale = 0.35
    new_h = int(scale * h) if int(scale * h) > min_width else min_width
    new_w = int(scale * w) if int(scale * w) > min_width else min_width
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


#def main(orig_path, resized_path, hocr_path, seg_path):
def main():
    args, _ = parse_inference_arguments()
    img_path = Path(args.image_path)
    checkpoints_path = Path(args.checkpoints)
    output_dir = Path(args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resized_img_name = str(img_path).split('/')[-1].split('.')[0] + '_resized' + '.jpg' 
    resized_path = Path(output_dir / resized_img_name)
    hocr_file = str(img_path).split('/')[-1].split('.')[0]
    hocr_path = Path(output_dir / hocr_file)
    segmented_img_name = str(img_path).split('/')[-1].split('.')[0] + '_segmented' + '.jpg' 
    segmented_img_path = Path(output_dir / segmented_img_name)
    col_segmented_img = str(img_path).split('/')[-1].split('.')[0] + '_col-segmented' + '.jpg'
    col_segmented_img_path = Path(output_dir / col_segmented_img)
    
    config_file = Path('../config.ini')
    reading_config(config_file)

    img = Image.open(img_path)
    img = resize_img(img)
    cv2.imwrite(str(resized_path), img)
    
    img_to_hocr(str(resized_path), str(hocr_path))


    split_model = SplitModel(input_channels=1)
    net = split_model.to(Config.get('device'))
    checkpoint = torch.load(checkpoints_path, map_location=Config.get('device'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    height, width = img.shape

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    transform = transforms.ToTensor()
    img = preprocess_img(img, transform)
    img = img.unsqueeze(1)

    pred_labels = net(img)
    pred_row_labels, pred_col_labels = pred_labels
    c5 = pred_col_labels[2]
    r5 = pred_row_labels[2]

    r5 = r5[-1] > 0.5
    c5 = c5[-1] > 0.5
    
    r5 = r5.cpu().detach().numpy()
    c5 = c5.cpu().detach().numpy()

    r_im = r5.reshape((-1,1))*np.ones((r5.shape[0],c5.shape[0]))
    c_im = c5.reshape((1,-1))*np.ones((height,width))
    im = cv2.bitwise_or(r_im,c_im)

    image = Image.fromarray(im*255.).convert('L')
    image.save(segmented_img_path)

    col_image = Image.fromarray(c_im*255).convert('L')
    col_image.save(col_segmented_img_path)

    hocr_file = str(hocr_path) + '.hocr'
    #df = extract_csv(segmented_img_path, hocr_file)
    df = extract_csv(col_segmented_img_path, hocr_file)
    print(df)

if __name__ == "__main__":
    main()
