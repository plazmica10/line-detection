import cv2
import numpy as np
def validate_input(min_line_len,max_line_gap):
    if min_line_len is None or min_line_len < 1:
        min_line_len = 10
    if max_line_gap is None or max_line_gap < 1:
        max_line_gap = 10
    return min_line_len,max_line_gap

def resize_image(img, max_size=1000):
    h,w = img.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(img, (new_w, new_h))

def connect_vertical_lines(vertical_lines,line_gap,plane_gap=5):
    new_vertical_lines = []
    for i in range(len(vertical_lines)):
        x0,y0,x1,y1 = vertical_lines[i].flatten()
        if x0 == 0:
            continue
        for j in range(i+1,len(vertical_lines)):
            x2,y2,x3,y3 = vertical_lines[j].flatten()
            #checking if they are the same line and close enough
            if abs(x0-x2) < plane_gap and (abs(y1-y2) < line_gap or abs(y0-y3) < line_gap or abs(y0-y2) < line_gap or abs(y1-y3) < line_gap):
                merged_line = [x0,min(y0,y2),x2,max(y1,y3)]
                vertical_lines[i] = merged_line
                vertical_lines[j] = [[0,0,0,0]]
        new_vertical_lines.append(vertical_lines[i])
    new_vertical_lines = np.array([line for line in new_vertical_lines if line[0][0] != 0])
    return new_vertical_lines
    
def connect_horizontal_lines(horizontal_lines,line_gap,plane_gap=5):
    new_horizontal_lines = []
    for i in range(len(horizontal_lines)):
        x0,y0,x1,y1 = horizontal_lines[i].flatten()
        if y0 == 0:
            continue
        for j in range(i+1,len(horizontal_lines)):
            x2,y2,x3,y3 = horizontal_lines[j].flatten()
            #checking if they are the same line and close enough
            if abs(y0-y2) < plane_gap and (abs(x1-x2) < line_gap or abs(x0-x3) < line_gap):
                merged_line = [min(x0,x2),y0,max(x1,x3),y2]
                horizontal_lines[i] = merged_line
                horizontal_lines[j] = [[0,0,0,0]]
        new_horizontal_lines.append(horizontal_lines[i])
    new_horizontal_lines = np.array([line for line in new_horizontal_lines if line[0][1] != 0])
    return new_horizontal_lines

def connect_perpendicular_lines(vertical_lines,horizontal_lines,line_gap):
    for i in range(len(horizontal_lines)):
        x0,y0,x1,y1 = horizontal_lines[i].flatten()
        for j in range(len(vertical_lines)):
            x2,y2,x3,y3 = vertical_lines[j].flatten()
            ###VERTICAL LINE IS ON THE LEFT###
            #horizontal start - vertical start
            if abs(x0-x2) < line_gap and abs(y0-y2) < line_gap:
                if x2 < x0:
                    x0 = x2
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y0 < y2:
                    y2 = y0
                    vertical_lines[j] = [[x2,y2,x3,y3]]

            #horizontal start - vertical end
            if abs(x0-x3) < line_gap and abs(y0-y3) < line_gap:
                if x3 < x0:
                    x0 = x3
                    horizontal_lines[i] = np.array([[x0, y0, x1, y1]])
                if y0 > y3:
                    y3 = y0
                    vertical_lines[j] = [[x2,y2,x3,y3]]

            #horizontal end - vertical start
            if abs(x1-x2) < line_gap and abs(y1-y2) < line_gap:
                if x2 < x1:
                    x1 = x2
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y1 < y2:
                    y2 = y1
                    vertical_lines[j] = [[x2,y2,x3,y3]]

            #horizontal end - vertical end
            if abs(x1-x3) < line_gap and abs(y1-y3) < line_gap:
                if x3 < x1:
                    x1 = x3
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y1 > y3:
                    y3 = y1
                    vertical_lines[j] = [[x2,y2,x3,y3]]
            ###VERTICAL LINE IS ON THE RIGHT###
            #horizontal end - vertical end
            if abs(x1-x3) < line_gap and abs(y1-y3) < line_gap:
                if x3 > x1:
                    x1 = x3
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y1 > y3:
                    y3 = y1
                    vertical_lines[j] = [[x2,y2,x3,y3]]  

            #horizontal end - vertical start
            if abs(x1-x2) < line_gap and abs(y1-y2) < line_gap:
                if x2 > x1:
                    x1 = x2
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y1 < y2:
                    y2 = y1
                    vertical_lines[j] = [[x2,y2,x3,y3]]  
            
            #horizontal start - vertical start
            if abs(x0-x2) < line_gap and abs(y0-y2) < line_gap:
                if x2 > x0:
                    x0 = x2
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y0 < y2:
                    y2 = y0
                    vertical_lines[j] = [[x2,y2,x3,y3]]

            #horizontal start - vertical end
            if abs(x0-x3) < line_gap and abs(y0-y3) < line_gap:
                if x3 > x0:
                    x0 = x3
                    horizontal_lines[i] = [[x0, y0, x1, y1]]
                if y0 > y3:
                    y3 = y0
                    vertical_lines[j] = [[x2,y2,x3,y3]]
    return vertical_lines,horizontal_lines