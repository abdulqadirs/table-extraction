import numpy as np
import cv2
import torch 
from PIL import Image
from config import Config
from pathlib import Path

from utils.read_config import reading_config
from models.split.split import SplitModel
#from preprocess import preproces_img
from postprocessing.ocr import img_to_hocr
from postprocessing.table_extraction import extract_csv
from utils.parse_arguments import parse_inference_arguments


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
    checkpints_path = Path(args.checkpoints)
    output_dir = Path(args.output_dir)

    
    # config_file = Path('../config.ini')
    # reading_config(config_file)

    # img = Image.open(orig_path)
    # img = resize_img(img)
    # cv2.imwrite(str(resized_path), img)
    # img_to_hocr(str(resized_path), str(hocr_path))


    # split_model = SplitModel(input_channels=1)
    # net = split_model.to(Config.get('device'))
    # checkpoint_file = Path('output/trained_model/checkpoint.deep_column_splitting.pth.tar')
    # checkpoint = torch.load(checkpoint_file, map_location=Config.get('device'))
    # net.load_state_dict(checkpoint['net'])
    # net.eval()
    # height, width = img.shape

    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # #img = preproces_img(img) ##todo
    # img = img.unsqueeze(1)
    # pred_labels = net(img)
    # c5 = pred_labels[2]
    # #r = r[-1] > 0.5
    # c5 = c5[-1] > 0.5
    # #r = r.cpu().detach().numpy()
    # c5 = c5.cpu().detach().numpy()
    
    # c_im = c5.reshape((1,-1))*np.ones((height, width))

    # image = Image.fromarray(c_im*255.).convert('L')
    # image.save(seg_path)
    # hocr_file = str(hocr_path) + '.hocr'
    # df = extract_csv(seg_path, hocr_file)
    # print(df)
    #return df

if __name__ == "__main__":
    main()
