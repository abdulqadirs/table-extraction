class EPS:
    def __init__(self):
        pass 
    
    def detect_eps_row(self, row):
        """
        Detect the earnings per share row.

        Args:
            row (list): List of strings.
        
        Returns:
            bool: True if the given row is earnings per share row and vice versa.
        """
        context = ['Earnings per share',
                'Earning per share',
                'earnings per share',
                'Earnings / (loss) per share',
                'Earnings / (Loss) per Ordinary share',
                'Earning / (Loss) per share - basic',
                'basic and diluted',
                'Earnings / (loss) per ordinary share — basic and diluted',
                'Distributable earnings per ordinary share',
                'Earnings per ordinary share',
                'earnings per ordinary share',
                'Basic and diluted earnings per share',
                'Basic and diluted (Rupees)',
                'Earnings/ (loss) per share - Basic and diluted (Rs)',
                'Earings/(loss) per share',
                'per share - basic and diluted',
                'Loss per share',
                'Loss per share - basic / diluted',
                'Earnings / (Loss) per share - basic',
                'Earnings (loss) per share - basic',
                '(Loss) / earnings per share - basic'
                '(Loss) / Earning per share']
        match = False
        for heading in context:
            if heading in row[0]:
                match = True
                break
        if row[0] in context or match == True:
            return True
        else:
            return False

    def decimal_conversion(self, numbers):
        """
        Adds a decimal point to the list of numbers, two decimal places from the RHS.

        Args:
            numbers (list): List of numbers(strings).
        
        Returns:
            new_numbers (list): List of numbers(strings) after decimal conversion.
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
