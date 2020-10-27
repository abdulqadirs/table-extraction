import pandas as pd

from postprocessing.bboxes import BBoxes
from postprocessing.midpoints import find_midpoints
from postprocessing.segmentation import TableSegmentation
from postprocessing.split_columns import split_merged_columns
from postprocessing.delete_columns import delete_duplicate_columns, delete_empty_columns
from postprocessing.merge_columns import merge_split_columns

def extract_csv(segmented_path, hocr_path):
    
    midpoints = find_midpoints(segmented_path)
    
    document_bboxes = BBoxes(hocr_path)
    document_bboxes.parse_hocr()
    document = document_bboxes.document_bboxes()

    table_segmentation = TableSegmentation(document)
    table, table_bboxes, _ =table_segmentation.column_segmentation(midpoints) 

    new_midpoints = split_merged_columns(table, table_bboxes, midpoints)
    new_midpoints.sort()

    table_segmentation1 = TableSegmentation(document)
    table, table_bboxes, _ = table_segmentation1.column_segmentation(new_midpoints)

    delete_empty_columns(table, table_bboxes)
    delete_duplicate_columns(table, table_bboxes)

    merge_split_columns(table, table_bboxes)
    delete_empty_columns(table, table_bboxes)

    columns = list(range(len(table)))
    df = pd.DataFrame(columns=columns)
    for i in columns:
        df[i] = table[i]

    return df