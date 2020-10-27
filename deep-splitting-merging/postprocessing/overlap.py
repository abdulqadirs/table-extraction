def horizontal_overlap(bbox1, bbox2):
    overlaps = True
    b1x1 = bbox1[0]
    b1x2 = bbox1[2]
    b2x1 = bbox2[0]
    b2x2 = bbox2[2]
    if (b1x2 < b2x1) or (b1x1 > b2x2):
        overlaps = False
    return overlaps