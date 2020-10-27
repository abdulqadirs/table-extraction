
def delete_empty_columns(table, table_bboxes):
    del_indices = []
    for i, col in enumerate(table):
        if len(col) == col.count(''):
            del_indices.append(i)
    
    for i, index in enumerate(del_indices):
        del table[index - i]
        del table_bboxes[index - i]

def delete_duplicate_columns(table, table_bboxes):
    del_indices = []
    for i in range(len(table) - 1):
        current_bbox = table_bboxes[i]
        next_bbox = table_bboxes[i+1]
        if current_bbox == next_bbox:
            del_indices.append(i)
    
    for i, index in enumerate(del_indices):
        del table[index - i]
        del table_bboxes[index - i]
        