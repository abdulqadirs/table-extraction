from image_csv.image_csv import ImageToCSV

def main():
    pdf_path = '../data/2019-q2.pdf'
    jpg_dir = '../data/'
    jpg_path = '../data/2019-q2.jpg'
    txt_path = '../data/2019-q2.txt'
    csv_path = '../data/out.csv'

    converter = ImageToCSV()
    #converter.pdf_to_jpg(pdf_path=pdf_path, jpg_dir=jpg_dir, page_no=None)
    #converter.image_to_txt(jpg_path, txt_path)
    converter.txt_to_csv(txt_path, csv_path)

if __name__ == "__main__":
    main()