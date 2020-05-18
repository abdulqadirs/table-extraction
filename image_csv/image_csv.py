from image_csv.data import remove_special_chars, decimal_conversion, find_year

from pdf2image import convert_from_path 
from PIL import Image 
import pandas as pd
import os
import re

class ImageToCSV:
    """
    """
    def __init__(self):
        pass 

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
            img = Image.open(img_path)
            width, height = img.size
            if  width >= 1000:
                left1 = 100
                upper1 = 20
                right1 = (width / 2) + 60
                lower1 = height - 200
                img1 = img.crop((left1, upper1, right1, lower1))

                left2 = right1
                upper2 = 20
                right2 = width
                lower2 = height - 200
                img2 = img.crop((left2, upper2, right2, lower2))

                path1 = str(img_path).split('.jpg')[0] + '-' + str(1) + '.jpg'
                path2 = str(img_path).split('.jpg')[0] + '-' + str(2) + '.jpg'

                img1.save(path1)
                img2.save(path2)

                return [path1, path2]

            else:
                return [img_path]
        else:
            print('FileNotFound')
    
    def pdf_to_jpg(self, pdf_path, jpg_dir, page_no=None):
        """
        Converts a specific page from  pdf from to  image(jpg). 
        If 'page_no' is not then all the pages from pdf files are converted to image(jpg).

        Args:
            pdf_path (Path): Path  of the pdf file.
            jpg_path (Path): Path of the output directory.
            page_no (int): Page number.
        
        Returns:
            Saves the jpg file(s) in the given directory.
        
        Raises:
            FileNotFoundError: If the given paths don't exist.
        """
        if os.path.exists(pdf_path):
            if page_no:
                jpg_file_name = str(pdf_path).split('/')[-1].split('.')[0] + '.jpg'
                jpg_file_path = str(jpg_dir) + '/' + jpg_file_name
                page = convert_from_path(pdf_path, first_page=page_no, last_page=page_no, grayscale=False)
                page[0].save(jpg_file_path, 'JPEG')
            else:
                pages = convert_from_path(pdf_path, grayscale=False)
                for i, page in enumerate(pages, start=1):
                    jpg_file_name = str(pdf_path).split('/')[-1].split('.')[0] + '-' + str(i) + '.jpg'
                    jpg_file_path = str(jpg_dir) + '/' + jpg_file_name
                    page.save(jpg_file_path, 'JPEG')
        else:
            print('FileNotFound')
        
    def image_to_txt(self, img_path, txt_dir):
        """
        Converts a given image to txt ffile.

        Args:
            img_path (Path): Path of the input image file.
            txt_path (Path): Directory of the output txt file.

        Returns:
            Saves the output txt file in the given directory

        Raises:
            FileNotFoundError: If the given paths don't exist.

        """
        if os.path.exists(img_path):
            paths = self.crop_image(img_path)
            for path in paths:
                txt_file_name = str(path).split('/')[-1].split('.')[0]
                txt_file_path = str(txt_dir) + '/' + txt_file_name
                cmd = 'tesseract ' + str(path) + ' ' + str(txt_file_path) + ' --psm 6 --dpi 300'
                os.system(cmd)
        else:
            print('FileNotFound')


    def txt_to_csv(self, txt_path, csv_path):
        """
        Converts a given text file to csv file.

        Args:
            txt_path (Path): Path of the input txt file.
            csv_path (Path): Path of the output csv file.
        
        Returns:
            Saves the output csv file in the given directory.
        
        Raises:
            FileNotFoundError: If the given paths don't exist.
        """
        if os.path.exists(txt_path):
            file = open(txt_path, 'r')
            regex = r'^[+-]{0,1}((\d*\.)|\d*)\d+$'
            note = False
            earnings = False
            start = False
            data = []
            max_columns = 0
            for line in file:
                line = line.split(' ')
                row = []
                headings = []
                numbers = []
                for word in line:
                    word = remove_special_chars(word)
                    if 'Note' in word or word == 'Note' or 'Nate' in word:
                        note = True
                    if 'annexed' in line:
                        break
                    if re.match(regex, word):
                        if earnings == True:
                            word = decimal_conversion(word)
                        numbers.append(word)
                    else:
                        headings.append(word)
                if headings:
                    headings = ' '.join(headings)
                    row.append(headings)

                if numbers:
                    num = numbers[0].replace('-', '')
                    num = num.replace('.', '')
                    if num.isdigit():
                        note_no = int(num)
                        if note_no < 100 and note == True:
                            del numbers[0]
                        row.extend(numbers)
                
                year = find_year(row)
                if year:
                    row.insert(0, 'Year')
                    start = True
                
                if row != [] and row != [''] and row is not None:
                    if 'Earnings per share' in row[0] or 'Earnings' in row[0] or 'per share' in row[0]:
                        earnings = True
                        row = decimal_conversion(row)
                #row = delete_row(row)
                if (row != [''] or row != []) and start == True:
                    #print(row)
                    if len(row) >= max_columns:
                        max_columns = len(row)
                    data.append(row)
                if earnings:
                    break
            
            columns = list(range(max_columns))
            for i, row in enumerate(data):
                if len(row) < max_columns:
                    data[i] = data[i] + [None] * (max_columns - len(row))
            df = pd.DataFrame(data, columns = columns)    
            print(df)
            df.to_csv(csv_path, sep='\t',index=False)