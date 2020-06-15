import logging
import os
import re
import sys

from image_csv.data import DataCleaner, detect_numbers
from image_csv.earnings_per_share import EPS
from image_csv.split import split_row
from PyPDF2 import PdfFileReader

from pdf2image import convert_from_path 
from pdf2image.exceptions import (PDFInfoNotInstalledError,PDFPageCountError,PDFSyntaxError)
from pathlib import Path
from PIL import Image 
import pandas as pd

#sys.tracebacklimit = 0
logger = logging.getLogger("table-extraction")

class PDFToCSV:
    """
    Takes a pdf and converts the given pages to jpg.
    The jpg images are then converted to txt.
    The txt file is finally converted to csv file.

    Args:
        input_path (Path): The path of the input pdf.
        output_path (Path): The path of the output directory to store csv file(s).
        ocr_path (Path): The path of the finetuned ocr directory.
        page_numbers (list): A list of the page number(s).
    """
    def __init__(self, input_path, output_path, ocr_path, page_numbers, double_page, next_page):
        self.input_path = input_path
        self.output_dir = output_path
        self.ocr_path = '/home/abdulqadir/tesseract/data/output'
        self.page_numbers = page_numbers
        self.double_page = double_page
        self.next_page = next_page
        self.image_files = []
        self.csv_files = []
        self.txt_files = []
        self.data_cleaner = DataCleaner()
        self.eps = EPS()

    def crop_image(self, img_path):
        """
        If the image has two columns then it crops the image into two images.

        Args:
            img_path (Path): Path of the image.
        
        Returns:
            Paths of the cropped images or None

        Raises:
            FileNotFoundError: If the given path doesn't exist. 
        """
        if os.path.exists(img_path):
            logger.info("Divding the double page image into two images.")
            img = Image.open(img_path)
            width, height = img.size
            left1 = 20
            upper1 = 20
            right1 = (width / 2)
            lower1 = height - 100
            img1 = img.crop((left1, upper1, right1, lower1))

            left2 = right1
            upper2 = 20
            right2 = width
            lower2 = height - 100
            img2 = img.crop((left2, upper2, right2, lower2))

            path1 = str(img_path).split('.jpg')[0] + '-' + str(1) + '.jpg'
            path2 = str(img_path).split('.jpg')[0] + '-' + str(2) + '.jpg'

            img1.save(path1)
            img2.save(path2)

            return [path1, path2]

        else:
            logger.exception("File %s not found.", img_path)
    
    def pdf_to_jpg(self):
        """
        Converts specific page(s) from  pdf from to  image(jpg). 
        Saves the output txt file in the given directory

        Args:
        
        Returns:
        
        Raises:
            FileNotFoundError: If the given paths don't exist.
        """
        if os.path.exists(self.input_path) and os.path.exists(self.output_dir):
            try:
                pdf = PdfFileReader(open(self.input_path, 'rb'))
                total_pages = pdf.getNumPages()
            except:
                logger.exception("Invalid pdf file path %s", self.input_path)
                sys.exit()
            else:
                for page_no in self.page_numbers:
                    if page_no > 0 and page_no <= total_pages:
                        jpg_file_name = str(self.input_path).split('/')[-1].split('.')[0]
                        jpg_file_path = str(self.output_dir) + '/' + jpg_file_name + '-' + str(page_no) + '.jpg'
                        page = convert_from_path(self.input_path, first_page=page_no, last_page=page_no, grayscale=False)
                        if len(page) != 0:
                            page[0].save(jpg_file_path, 'JPEG')
                            self.image_files.append(jpg_file_path)
                        else:
                            logger.exception("Can't convert the given page of pdf file to jpg.")
                            sys.exit()
                    else:
                        logger.exception("Enter the correct page number.")
                        sys.exit()
        else:
            if os.path.exists(self.input_path) is False and os.path.exists(self.output_dir) is False:
                logger.exception("Input file %s and output directory %s not found!", self.input_path, self.output_dir)
                sys.exit()
            elif os.path.exists(self.input_path):
                logger.exception("Ouput directory %s not found!", self.output_dir)
                sys.exit()
            elif os.path.exists(self.output_dir):
                logger.exception("Input file %s not found!", self.input_path)
                sys.exit()
            

    def image_to_txt(self):
        """
        Converts a given image to txt file.
        Saves the output txt file in the given directory

        Args:

        Returns:

        Raises:
            FileNotFoundError: If the given paths don't exist.
        """
        if os.path.exists(self.output_dir):
            for image_file in self.image_files:
                if self.double_page:
                    paths = self.crop_image(image_file)
                    for path in paths:
                        txt_file_name = str(path).split('/')[-1].split('.')[0]
                        txt_path = str(self.output_dir) + '/' + txt_file_name
                        cmd = 'tesseract --tessdata-dir ' + self.ocr_path + ' ' + path + ' ' + txt_path + ' --psm 6'
                        os.system(cmd)
                        self.txt_files.append(txt_path + '.txt')
                else:
                    txt_file_name = str(image_file).split('/')[-1].split('.')[0]
                    txt_path = str(self.output_dir) + '/' + txt_file_name
                    cmd = 'tesseract --tessdata-dir ' + self.ocr_path + ' ' + image_file + ' ' + txt_path + ' --psm 6'
                    os.system(cmd)
                    self.txt_files.append(txt_path + '.txt')
        else:
            print('FileNotFound')


    def txt_to_csv(self):
        """
        Converts a given text file to csv file.
        Saves the output csv file in the given directory.

        Args:
        
        Returns:
        
        Raises:
        """
        for txt_file in self.txt_files:
            file = open(txt_file, "r")
            earnings = False
            data = []
            max_columns = 0
            earnings_row = None
            earnings = False
            earn_prev = False
            columns = 10 * [0]
            right_cols = []
            left_cols = [] 
            for line_no, line in enumerate(file):
                line = line.split(' ')
                row = []
                heading = False
                left = False
                right = False
                left_numbers, headings, right_numbers = split_row(line)
                if left_numbers == right_numbers and not self.data_cleaner.isempty_row(left_numbers):
                    if left_cols is not None and len(left_cols) > 4:
                        left_numbers = left_numbers[0:len(left_numbers) // 2]
                        right_numbers = right_numbers[len(right_numbers) // 2 :]
                    else:
                        left_numbers = []

                if not self.data_cleaner.isempty_row(left_numbers):
                    #left_numbers = delete_note(left_numbers)
                    row.extend(left_numbers)
                    left = True
                    left_cols.append(len(left_numbers))
                    
                if not self.data_cleaner.isempty_row(headings):
                    headings = ' '.join(headings)
                    headings = self.data_cleaner.clean_heading(headings)
                    row.append(headings)
                    heading = True
                else:
                    headings = ' '
                    row.append(headings)

                if not self.data_cleaner.isempty_row(right_numbers):
                    right_numbers = self.data_cleaner.delete_note(right_numbers)
                    row.extend(right_numbers)
                    right = True
                    right_cols.append(len(right_numbers))
                
                if left is False and right is False and heading is  True:
                    if left_cols != [] and len(left_cols) > 3:
                        index = max(set(left_cols), key=left_cols.count)
                        if index > 0:
                            row = [''] * index + row
                        
                if not self.data_cleaner.isempty_row(row) and earnings is False:
                    earnings = self.eps.detect_eps_row(row)
                if earnings:
                    if len(row) == 1:
                        earnings_row = line_no
                        earn_prev = True
                    else:
                        row = self.eps.decimal_conversion(row)

                if earn_prev == True and line_no == (earnings_row + 1):
                    row = self.eps.decimal_conversion(row)
                    earn_prev = False
                        
                if not self.data_cleaner.isempty_row(row):
                    if len(row) >= max_columns:
                        max_columns = len(row)
                    if not self.data_cleaner.delete_line(row):
                        data.append(row)
                        #print(row)
                        columns[len(row)] += 1

            columns = list(range(max_columns))
            for i, row in enumerate(data):
                all_nums = detect_numbers(row)
                if all_nums:
                    data[i] = [None] * (max_columns - len(row)) + data[i]
                else:
                    data[i] = data[i] + [None] * (max_columns - len(row))
                 
            df = pd.DataFrame(data, columns = columns)

            csv_file = txt_file.split('.txt')[0] + '.csv'
            df.to_csv(csv_file, sep='\t',index=False)
            self.csv_files.append(csv_file)
    

    def combine_csv_files(self):
        """
        If the financial report extends to the next page then the csv files of the two page are combined.

        """
        if self.next_page:
            file1 = self.csv_files[0]
            file2 = self.csv_files[1]
            ##todo check if the file ends with .pdf
            file_name = str(self.input_path).split('/')[-1].split('.')[0]
            output_file = str(self.output_dir) + '/' + file_name + '.csv'
            df1 = pd.read_csv(file1, delimiter = '\t')
            df2 = pd.read_csv(file2, delimiter = '\t')
            df = df1.append(df2)
            df.to_csv(output_file, sep='\t',index=False)