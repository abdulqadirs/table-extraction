
class TableSegmentation:
    def __init__(self, document):
        self.document = document
        self.table = []
        self.table_bboxes = []
        self.overlapping_cells = []

    def column_segmentation(self, mid_points):
        for i,midpoint in enumerate(mid_points):
            column = []
            last_column = []
            column_bboxes = []
            last_column_bboxes = []
            for j, line in enumerate(self.document):
                col_words = ''
                last_col_words = ''
                col_bboxes = []
                last_col_bboxes = []
                words = self.document[line]['words']
                bboxes = self.document[line]['bboxes']
                for k, line_data in enumerate(zip(words, bboxes)):
                    word, bbox = line_data
                    start = bbox[0]
                    end = bbox[2]
                    if i == 0:
                        if midpoint >= end:
                            col_words = " ".join((col_words, word))
                            col_bboxes.append(bbox)
                        #overlapping cells
                        if start < midpoint and end > midpoint:
                            col_words = " ".join((col_words, word))
                            col_bboxes.append(bbox)
                            self.overlapping_cells.append((j, i))
                        
                        if len(mid_points) == 1:
                            if start > midpoint and end > midpoint:
                                last_col_words = " ".join((last_col_words, word))
                                last_col_bboxes.append(bbox)
                            
                    elif i == len(mid_points) - 1:
                        if start >= mid_points[i-1] and end <= mid_points[i]:
                            col_words = " ".join((col_words, word))
                            col_bboxes.append(bbox)
                        #overlapping cells
                        if start < midpoint and end > midpoint:
                            col_words = " ".join((col_words, word))
                            col_bboxes.append(bbox)
                            self.overlapping_cells.append((j, i))
                        if midpoint <= start and midpoint < end:
                            last_col_words = " ".join((last_col_words, word))
                            last_col_bboxes.append(bbox)
                    else:
                        if start > mid_points[i-1] and end <= mid_points[i]:
                            col_words = " ".join((col_words, word))
                            col_bboxes.append(bbox)
                        #overlapping cells
                        if start < midpoint and end > midpoint:
                            col_words = " ".join((col_words, word))
                            col_bboxes.append(bbox)
                            self.overlapping_cells.append((j, i))
                        
                            
                column.append(col_words.strip())
                column_bboxes.append(col_bboxes)
                if i == len(mid_points) - 1:
                    last_column.append(last_col_words.strip())
                    last_column_bboxes.append(last_col_bboxes)
                    
            self.table.append(column)
            self.table_bboxes.append(column_bboxes)
            if last_column != []:
                self.table.append(last_column)
                self.table_bboxes.append(last_column_bboxes)
        return self.table, self.table_bboxes, self.overlapping_cells