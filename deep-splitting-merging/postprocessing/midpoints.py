import numpy as np
from PIL import Image

def find_midpoints(path):
    img = np.asarray(Image.open(path))
    col_segmentation = img[0]
    sep_indices = np.where(col_segmentation == 255)
    diff = np.where(np.diff(sep_indices) > 5)[1]
    diff = [i + 1 for i in diff ]
    
    separators = np.split(sep_indices[0], diff)
    #print(separators)
    sep_regions = []
    for separator in separators:
        #print(separator)
        start = separator[0]
        end = separator[-1]
        sep_regions.append((start, end))
    
    mid_points = []
    for sep_region in sep_regions:
        mid_point = int(np.floor((sep_region[0] + sep_region[1]) / 2))
        mid_points.append(mid_point)
    
    return mid_points
