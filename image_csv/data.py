import re

class DataCleaner:
    def __init__(self):
        pass

    def clean_row(self, row):
        """
        Delete unnecessary words from the given row.

        Args:
            row (list): A list of strings.
        
        Returns:
            row (list): A list of strings.
        """
        if row[-1] == ':\n' or row[-1] == '\n':
            row = row[0:len(row)-1]
        for i, word in enumerate(row):
            if word == 'Rs' or 'Rs' in word:
                del row[i]
        return row

    def clean_heading(self, heading):
        """
        Delete special characters from the table row headings.

        Args:
            heading (string):
        
        Returns:
            heading (string):
        """
        heading = heading.replace('\n', '')
        heading = heading.replace('\t', ' ')
        return heading

    def clean_number(self, number):
        """
        Removes all the special characters from the given string (number).
        If the numbers are surrounded by round bracket '()' it replaces them with negivtive sign '-'.

        Args:
            number (str):
        
        Returns: 
            number (string): 
        """
        number = number.replace(':\n', '')
        number = number.replace(' ','')
        number = number.replace(',', '')        
        number = number.replace('\n', '')
        number = number.replace('|', '')
        number = number.replace('[', '')
        number = number.replace('.', '')

        number = number.replace(']', '')
        number = number.replace('!', '')
        number = number.replace('-Rs', '')
        number = number.replace('Rs', '')
        #if number.startswith('('):
        #    number = number.replace('(', '-', 1)
        number = number.replace('(', '')
            
        if not number.startswith('-') and number.endswith(')'):
            number = number.replace(')', '')
            number = '-' + number
        number = number.replace(')', '')

        if number.startswith('(') and number.endswith(')'):
            number = '-' + number
            number = number.replace('(', '')
            number = number.replace(')', '')
            
        if number.endswith(')') and not number.startswith('('):
            number = number.replace(')', '')
            
        if not number.endswith(')') and number.startswith('('):
            number = number.replace('(', '')  
        return number


    def delete_note(self, numbers):
        """
        Deleting the note column.

        Args: 
            numbers (list): List of strings.
        
        Returns:
            numbers (list): List of strings after deleting 'note' column.
        """
        index = 0
        num = numbers[index].strip()
        if num == '-':
            return numbers
        elif float(numbers[index]) < 250 and float(numbers[index]) > 0:
            del numbers[0]
        return numbers     
    
    def delete_line(self, row):
        """
        Delete the desired row.
        """
        if row == 'CONTINGENCIES AND COMMITMENTS' or 'CONTINGENCIES AND COMMITMENTS' in row or 'Contingencies and commitments' in row:
            return True
        else:
            return False
    
    def isempty_row(self, row):
        """
        Detects if a row is empty or not.

        Args:
            row (list):
        
        Returns:
            bool:

        """
        if row == [] or row == [''] or row is None or row == [' ']:
            return True
        else:
            return False


def detect_numbers(row):
    """
    Returns true if all the strings in a given row are numbers.

    Args:
        row (list): List of strings.
    
    Returns:
        row (list): 
    """
    count = 0
    for num in row:
        if num.startswith('-') and num != '-':
            num = num.replace('-', '')
        if num is not None:
            if num.isdigit():
                count = count + 1
    if count == 0:
        return False
    elif count == len(row):
        return True
    else:
        return False