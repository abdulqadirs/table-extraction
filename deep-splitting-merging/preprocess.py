import numpy as np
import cv2
from PIL import Image, ImageChops 
import PIL.ImageOps
import xml.etree.ElementTree as ET
from typing import Tuple, Union
import math
#from deskew import determine_skew
import time
import os
import json
from pathlib import Path
from tqdm import tqdm
from utils.parse_arguments import parse_preprocessing_arguments


def read_xml(path, img):
    tree = ET.parse(path)
    root = tree.getroot()
    table_attrib = None
    rows = None
    columns = None
    col_percent = 98.5
    row_percent = 97
    tables = root.findall('Tables')
    for table in tables:
        for element in table.iter('Table'):
            table_attrib = element.attrib
            rows = element.findall('Row')
            columns = element.findall('Column')
    
    
    lines = []
    for row in rows:
        x0 = int(row.attrib['x0'])
        y0 = int(row.attrib['y0'])
        x1 = int(row.attrib['x1'])
        y1 = int(row.attrib['y1'])
        line = [x0, y0, x1, y1]
        lines.append(line)

    cols = []
    for column in columns:
        x0 = int(column.attrib['x0'])
        y0 = int(column.attrib['y0'])
        x1 = int(column.attrib['x1'])
        y1 = int(column.attrib['y1'])
        col = [x0, y0, x1, y1]
        cols.append(col)
    
    img_array = np.array(img)
    height = img_array.shape[0]
    width = img_array.shape[1]
    row_labels = np.zeros(height)
    col_labels = np.zeros(width)
#     if height < 600:
#         col_percent = 80
    for line in lines:
        row = line[1]
        col_start = line[0]
        col_end = line[2]
        separator = img_array[row][col_start:col_end]
        row_labels[row] = 255
        prev_end = True
        next_end = True
        row_count = 0
        for i in range(1,150):
            if (row - i) <= 0:
                continue
            elif (row + i) >= (height - 1):
                continue
            elif (row - i) <= 0 and (row + i) >= (height - 1):
                break
            previous_row = img_array[row - i][col_start:col_end]
            next_row = img_array[row + i][col_start:col_end]
            prev_count = (previous_row >= 200).sum()
            next_count = (next_row >= 200).sum()
            row_length = col_end - col_start
            prev_percentage = ((prev_count / row_length) * 100)
            next_percentage = ((next_count / row_length) * 100)
            if prev_percentage >= row_percent and prev_end == True and (row - i) > 1:
                row_labels[row - i] = 255
                row_count += 1
            else:
                prev_end = False
            if next_percentage >= row_percent and next_end == True and (row + i) < height:
                row_labels[row + i] = 255
                row_count += 1
            else:
                next_end = False
        if row_count < 10:
            for i in range(1,5):
                if row - i > 1:
                    row_labels[row - i] = 255
                elif row + i < height:
                    row_labels[row + i] = 255
                
    for column in cols:
        col = column[0]
        row_start = column[1]
        row_end = column[3]
        separator = img_array[:,col][row_start:row_end]
        col_labels[col] = 255
        prev_end = True
        next_end = True
        count = 0
        for i in range(1,150):
            if (col - i) <= 0:
                continue
            elif (col + i) >= (width - 1):
                continue
            elif (col - i) <= 0 and (col + i) >= (width - 1):
                break
            previous_col = img_array[:,col - i][row_start:row_end]
            next_col = img_array[:,col + i][row_start:row_end]
            prev_count = (previous_col >= 200).sum()
            next_count = (next_col >= 200).sum()
            col_length = row_end - row_start
            prev_percentage = ((prev_count / col_length) * 100)
            next_percentage = ((next_count / col_length) * 100)
            if prev_percentage >= col_percent and prev_end == True and (col - i) > 0:
                col_labels[col - i] = 255
                count += 1
            else:
                prev_end = False
            if next_percentage >= col_percent and next_end == True and (col + i) < width:
                col_labels[col + i] = 255
                count += 1
            else:
                next_end = False

        if count <= 28:
            for i in range(1,15):
                if (col - i) > 0:
                    col_labels[col - i] = 255
                elif (col + i) < width:
                    col_labels[col + i] = 255
        
    
    left = int(table_attrib['x0'])
    top = int(table_attrib['y0'])
    right = int(table_attrib['x1'])
    bottom = int(table_attrib['y1'])
    
    img = img.crop((left, top, right, bottom))
    row_labels = row_labels[top:bottom]
    col_labels = col_labels[left:right]

    return img, row_labels, col_labels

def remove_lines(orig_image, binary_img):
    src = orig_image
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
        
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 13, -2)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)
   
    cols = horizontal.shape[1]
    horizontal_size = cols // 30

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    #verticalsize = rows // 40
    verticalsize = 35

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)



    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (39,3))
    horz_dilated = cv2.dilate(horizontal, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
    vert_dilated = cv2.dilate(vertical, kernel, iterations=1)
    
    combined = cv2.add(horz_dilated,vert_dilated)
    
    
    combined = cv2.add(binary_img,combined)
    
    return combined


def binarize_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13,13)
    return img


def preprocess_data(input_images_dir, xml_dir, processed_images_dir, json_dir):
    image_files = os.listdir(input_images_dir)
    for img_file in tqdm(image_files):
        img_path = Path(input_images_dir /  img_file)
        xml_file = img_file.split('.')[0] + '.xml'
        xml_path = Path(xml_dir / xml_file)

        img = Image.open(img_path)
        #converting from pil to cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #img = deskew_image(img)
        bin_img = binarize_img(img)
        
        img = remove_lines(img, bin_img)
        #converting from cv2 to pil
        img = Image.fromarray(img)
        img, row_labels, col_labels = read_xml(xml_path, img)

        row_labels = (row_labels / 255).tolist()
        col_labels = (col_labels / 255).tolist()
        
        labels = {'rows' : row_labels, 'columns' : col_labels}

        json_file = img_file.split('.')[0] + '.json'

        json_path = Path(json_dir / json_file)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
        
        
        img_path = Path(processed_images_dir  /  img_file)

        img.save(img_path)


def preprocess_img(img, transform):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bin_img = binarize_img(img)
    img = remove_lines(img, bin_img)
    img = Image.fromarray(img)

    if transform is not None:
        img = transform(img)

    return img

def main():
    #parsing the arguments
    args, _ = parse_preprocessing_arguments()
    input_images_dir = Path(args.input_images)
    xml_dir = Path(args.xml_labels)
    processed_images_dir = Path(args.processed_images)
    json_dir = Path(args.json_labels)

    preprocess_data(input_images_dir, xml_dir, processed_images_dir, json_dir)

if __name__ == "__main__":
    main()