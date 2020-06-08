import re
from image_csv.data import DataCleaner

def split_row(row):
    """
    Splits a given row of table in headings and numbers.

    Args:
        row (list): List of strings.
    
    Returns:
        left_numbers (list): List of numbers(strings) before the heading.
        headings (list): List containing the heading.
        right_numbers (list): List of numbers (strings) after the heading.
    """
    data_cleaner = DataCleaner()
    regex = r'^[+-]{0,1}((\d*\.)|\d*)\d+$'
    headings = []
    right_numbers = []
    left_numbers = []
    i = 0
    j = 0
    row = data_cleaner.clean_row(row)
    for word in row:
        word = data_cleaner.clean_number(word)
        if re.match(regex, word):
            left_numbers.append(word)
            j += 1
        elif word == '-' or word == ' -' or word == '-':
            left_numbers.append(word)
            j += 1
        else:
            break
            
    for word in reversed(row):
        word = data_cleaner.clean_number(word)
        if re.match(regex, word):
            right_numbers.append(word)
            i += 1
        elif word == '-' or word == ' -' or word == '- ':
            right_numbers.append(word)
            i += 1
        else:
            break
    if len(left_numbers) < 2:
        left_numbers = []
    right_numbers.reverse()
    headings = row[len(left_numbers):len(row) - i]
    return left_numbers,headings,right_numbers