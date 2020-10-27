def merge_split_columns(table, table_bboxes):
    for i in range(len(table) - 1):
        current_col, current_bbox = table[i], table_bboxes[i]
        next_col, next_bbox = table[i+1], table_bboxes[i+1]
        current_next = False
        for j in range(len(current_bbox)):
            current_line, current_line_bbox = current_col[j], current_bbox[j]
            next_line, next_line_bbox = next_col[j], next_bbox[j]
            line_gap = None
            if current_line_bbox != [] and next_line_bbox != []:
                line_gap = next_line_bbox[0][0] - current_line_bbox[-1][2]
            else:
                continue
            
            if line_gap < 12 and current_col.count('') < next_col.count(''):
                table[i][j] = current_line + ' ' + next_line
                table[i+1][j] = ''
                table_bboxes[i][j].extend(next_line_bbox)
                table_bboxes[i+1][j] = []
                current_next = True
            elif line_gap < 12 and current_col.count('') >= next_col.count(''): 
                table[i+1][j] = current_line + ' ' + next_line
                table[i][j] = ''
                table_bboxes[i+1][j].extend(next_line_bbox)
                table_bboxes[i][j] = []
                current_next = False
                
        if table[i+1].count('') == len(next_col) and current_next == True:
            temp = table[i+1]
            table[i+1] = table[i]
            table[i] = temp
            
            temp_bboxes = table_bboxes[i+1]
            table_bboxes[i+1] = table_bboxes[i]
            table_bboxes[i] = temp_bboxes
        elif table[i+1].count('') == len(next_col) and current_next == False:
            pass
    