from bs4 import BeautifulSoup

class BBoxes:
    def __init__(self, path):
        self.path = path
        self.words = None
        self.lines = None
        self.segments = None
        self.all_words = []
        self.words_bbox = []
        self.document = {}
        
    def parse_hocr(self):
        soup = BeautifulSoup(open(self.path), "html.parser")
        self.lines = soup.find_all('span', {'class' : 'ocr_line'})
        self.words = soup.find_all('span', {'class' : 'ocrx_word'})
        self.segments = soup.find_all('div', {'class': 'ocr_carea'})
    
    def word_bboxes(self):
        for word in self.words:
            bbox = word['title'].split(';')[0].split(' ')[1:]
            bbox = [int(x) for x in bbox]
            special_chars = '!"#$%&()*+,.:;<=>?@[\]^_`{|}~»¢®$é“¢'
            special_words = ['||', '&.', '__', '|.']
            if word.text not in special_chars and word.text not in special_words:
                self.all_words.append(word.text)
                self.words_bbox.append(bbox)
        return self.all_words, self.words_bbox
    
    def document_bboxes(self):
        for s in self.segments:
            lines = s.find_all('span', {'class' : 'ocr_line'})
            line_bbox = {}
            for line in lines:
                bbox = line['title'].split(';')[0].split(' ')[1:]
                bbox = [int(x) for x in bbox]
                line_bbox[line.text.replace('\n', ' ')] = bbox
            
                words = line.find_all('span', {'class' : 'ocrx_word'})
                content = []
                content_bbox = []
                for word in words:
                    bbox = word['title'].split(';')[0].split(' ')[1:]
                    bbox = [int(x) for x in bbox]
                    special_chars = '!"#$%&()*+,.:;<=>?@[\]^_`{|}~»'
                    special_words = ['||', '&.', '__ ']
                    if word.text not in special_chars and word.text not in special_words:
                        content.append(word.text)
                        content_bbox.append(bbox)
                
                line_text = line.text.replace('\n', ' ').strip()
                if line_text == '':
                    continue
                else:
                    self.document[line_text] = {'words' : content, 'bboxes' : content_bbox}
        return self.document