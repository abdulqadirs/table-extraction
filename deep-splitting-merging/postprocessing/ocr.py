import os
import sys

def img_to_hocr(img_path, hocr_path):
    #tesseract file.tif output --psm 6 -c tessedit_create_hocr=1 -c tessedit_pageseg_mode=6
    cmd = 'tesseract ' + img_path + ' ' + hocr_path + ' --psm 6 -c tessedit_create_hocr=1 -c tessedit_pageseg_mode=6'
    os.system(cmd)
