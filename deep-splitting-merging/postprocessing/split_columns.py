
def split_merged_columns(table, table_bboxes, mid_points):
    old_midpoints = [mid_point for mid_point in mid_points]
    for word_col, bbox_col in zip(table, table_bboxes):
        centers = {}
        centers_distance = {}
        centers_count = {}
        for i in range(1, 10):
            centers[i] = []
            centers_distance[i] = []
            centers_count[i] = 0
        for i in range(len(word_col)):
            current_line, current_bboxes = word_col[i].replace('  ', ' ').split(' '), bbox_col[i]
            if len(current_bboxes) > 1:
                distances = []
                midpoints = []
                for j in range(len(current_bboxes) - 1):
                    current_end = current_bboxes[j][2]
                    next_start = current_bboxes[j + 1][0]
                    distance = int(next_start - current_end)
                    if distance > 15:
                        midpoints.append((current_end + next_start) // 2)
                        distances.append(distance)
                        #midpoints.append(next_start)

                if midpoints != []:
                    if centers[len(midpoints)] == []:
                        centers[len(midpoints)] = midpoints
                        centers_count[len(midpoints)] += 1
                        centers_distance[len(midpoints)] = distances
                    else:

                        temp = centers[len(midpoints)]
                        temp_dist = centers_distance[len(midpoints)]
                        #avg_result = [sum(x) // 2 for x in zip(temp, midpoints)]
                        avg_result = []
                        dist = []
                        for tm, m, d, td in zip(temp, midpoints, distances, temp_dist):
                            if d <= td:
                                avg_result.append(m)
                                dist.append(d)
                            else:
                                avg_result.append(tm)
                                dist.append(td)
                                
                        centers[len(midpoints)] = avg_result
                        centers_distance[len(midpoints)] = dist
                        centers_count[len(midpoints)] += 1
                #print(current_line)
                #print(current_bboxes)
                #print(distances)
        new_midpoints = None
        for c in centers_count:
            if centers_count[c] >= int(len(word_col) * 0.20):
                new_midpoints = centers[c]
        if new_midpoints is not None:
            old_midpoints.extend(new_midpoints)
        #print(centers)
        #print(centers_count)
        #print(new_midpoints)
        #print('------')
    return old_midpoints