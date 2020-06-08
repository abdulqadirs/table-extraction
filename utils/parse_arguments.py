from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

def parse_arguments():
    """
    Parses the arguments passed through the terminal.
    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--input_path', help='Path of the input pdf.', required=True)
    parser.add_argument('-o', '--output_path', help='Directory of the output csv file(s).', required=True)
    parser.add_argument('-t', '--finetuned_ocr', help='Directory where fineturned tesseract ocr is stored.', required=True)
    parser.add_argument('-b', '--balancesheet_page_nums', help='List containing balance sheet page number(s).', required=False)
    parser.add_argument('-p', '--profitloss_page_nums', help='List containing pofit-loss page number(s).', required=False)
    parser.add_argument('-d', '--double_page', help='Enter True if the pdf has double pages.', required=True)
    parser.add_argument('-n', '--next_page', help='Enter True if the financial statement extends to the next page.', required=True)

    return parser.parse_known_args()