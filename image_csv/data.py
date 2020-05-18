import re

def remove_special_chars(word):
    """
    Removes all the special characters from the given string.

    Args:
        word (str): String.
    """
    word = word.replace(',', '')
    if word.startswith('('):
        word = word.replace('(', '-', 1)
    word = word.replace('(', '')
        
    if not word.startswith('-') and word.endswith(')'):
        word = word.replace(')', '')
        word = '-' + word
    word = word.replace(')', '')  
    word = word.replace('\n', '')
    word = word.replace('|', '')
    word = word.replace('[', '')

    word = word.replace(']', '')
    word = word.replace('.', '')
    word = word.replace('!', '')
    word = word.replace('-Rs', '')
    return word

def decimal_conversion(numbers):
    """
    Converts the numbers to decimal.

    Args:
        numbers (list): List of strings(numbers).
    """
    new_numbers = []
    new_numbers.append(numbers[0])
    for num in numbers[1:]:
        if '.' not in num:
            l = len(num)
            end_sub = num[-2:l]
            start_sub = num.replace(end_sub, '')
            new_num = start_sub + '.' + end_sub
            new_numbers.append(new_num)
        elif ',' in num:
            num = num.replace(',', '.')
            new_numbers.append(num)
        elif '.' in num:
            new_numbers.append(num)
    return new_numbers

def find_quarter():
    """
    """
    pass 

def find_year(row):
    """
    Finds if years exist in the given string.

    Args:
        row (list): List of strings.
    """
    i = 0
    max_len = len(row)
    regex = r'[2][0][0-9]{2}'
    for word in row:
        if re.match(regex, word):
            i = i + 1
    if i == max_len and i > 1:
        return True
    else:
        return False