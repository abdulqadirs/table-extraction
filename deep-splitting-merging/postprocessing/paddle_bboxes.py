from paddleocr import PaddleOCR
import pandas as pd

class PaddleBBoxes:
    def __init__(self, img_path):
        self.img_path = img_path
        self.raw_data = None
        self.ocr()
        self.grouped_rows = [] #line bboxes of each word
        self.grouped_texts = [] #line text
        self.row_separators = [] # midpoints between rows along y-axis

    def percentage_vertical_overlap(self, bbox1, bbox2):
        # the percentage vertical overlap with respect to the minimum height rectangle.
        overlap_height = min(bbox1[2][1], bbox2[2][1]) - max(bbox1[0][1], bbox2[0][1])
        if overlap_height < 0:
            overlap_height = 0
        #print('ovelap height: ', overlap_height)
        h1 = bbox1[2][1] - bbox1[1][1]
        h2 = bbox2[2][1] - bbox2[1][1]
        minimum_height = min(h1, h2)
        if minimum_height == 0:
            return 0
        #print('minimum height: ', minimum_height)
        percent_vertical_overlap = (overlap_height / minimum_height) * 100
        return percent_vertical_overlap
    
    def polygon_to_rectangle(self, bbox):
        flatten_bbox = [coord for box in bbox for coord in box]
        min_x = min(coord[0] for coord in flatten_bbox)
        min_y = min(coord[1] for coord in flatten_bbox)
        max_x = max(coord[0] for coord in flatten_bbox)
        max_y = max(coord[1] for coord in flatten_bbox)
        rectangle = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        return rectangle
    
    def ocr(self):
         #data extraction using paddleocr
        ocr = PaddleOCR(lang='en')
        self.raw_data = ocr.ocr(self.img_path, cls=False)[0]
    
    def document_bboxes(self):
        #converting the ocr result to dataframe
        #in this data frame each row represents a single word(s) with bounding box
        df = pd.DataFrame(self.raw_data, columns=['bbox', 'text'])
        #grouping the words into rows(lines) based on percentage vertical overlap
        self.grouped_bboxes = []
        self.grouped_texts = []

        for index, row in df.iterrows():
            bbox = row['bbox']
            text = row['text']
            is_grouped = False
            
            for i, grouped_bbox in enumerate(self.grouped_bboxes):
                #if percentage vertical overlap is greater than 50% then group words in line
                if self.percentage_vertical_overlap(bbox, grouped_bbox[0]) > 50:
                    self.grouped_bboxes[i].append(bbox)
                    self.grouped_texts[i].append(text[0])
                    is_grouped = True
                    break
            
            if not is_grouped:
                self.grouped_bboxes.append([bbox])
                self.grouped_texts.append([text[0]])

        #document with format: {row_id : {'words' : txt, 'bboxes' : bbox}}
        document = {}
        for i, (txt, bboxes) in enumerate(zip(self.grouped_texts, self.grouped_bboxes)):
            rect_bboxes = []
            for bbox in bboxes:
                min_x = min(coord[0] for coord in bbox)
                min_y = min(coord[1] for coord in bbox)
                max_x = max(coord[0] for coord in bbox)
                max_y = max(coord[1] for coord in bbox)
                rect_bbox = [min_x, min_y, max_x, max_y]
                rect_bboxes.append(rect_bbox)
            document[i] = {'words' : txt, 'bboxes' : rect_bboxes}
        
        return document
    
    def line_separators(self):
        #converting word polygons of a line to line rectangle
        line_rectangles = []
        for bbox in self.grouped_bboxes:
            rectangle = self.polygon_to_rectangle(bbox)
            line_rectangles.append(rectangle)

        #finding midpoint between consecutive lines along y-axis
        for i in range(len(line_rectangles) - 1):
            curr_rect = line_rectangles[i]
            next_rect = line_rectangles[i + 1]

            y_avg = (curr_rect[2][1] + next_rect[0][1]) // 2
            self.row_separators.append(y_avg)

        return self.row_separators