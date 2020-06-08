import logging
from pathlib import Path

from image_csv.image_csv import PDFToCSV
from utils.parse_arguments import parse_arguments
from utils.setup_logging import setup_logging
from pathlib import Path

logger = logging.getLogger('table_extraction')

def main():
    args, _ = parse_arguments()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    pretrained_ocr = Path(args.finetuned_ocr)
    bsheet_pages = args.balancesheet_page_nums
    profitloss_pages = args.profitloss_page_nums
    page_numbers = []
    if bsheet_pages is not None:
        bsheet_pages = bsheet_pages.lstrip('[').rstrip(']').split(',')
        bsheet_pages = [int(p_no) for p_no in bsheet_pages]
        page_numbers.extend(bsheet_pages)
    if profitloss_pages is not None:
        profitloss_pages = profitloss_pages.lstrip('[').rstrip(']').split(',')
        profitloss_pages = [int(p_no) for p_no in profitloss_pages]
        page_numbers.extend(profitloss_pages)
    
    if bsheet_pages is None and profitloss_pages is None:
        print("Enter the page numbers.")
        return
    if args.double_page == 'True':
        double_page = True
    else:
        double_page = False
    
    if args.next_page == 'True':
        next_page = True
    else:
        next_page = False


    table_extractor = PDFToCSV(input_path, output_path, pretrained_ocr, page_numbers, double_page, next_page)
    table_extractor.pdf_to_jpg()
    table_extractor.image_to_txt()
    table_extractor.txt_to_csv()
    table_extractor.combine_csv_files()


if __name__ == "__main__":
    main()