import cv2
def validate_input(min_line_len,max_line_gap):
    if min_line_len is None or min_line_len < 1:
        min_line_len = 10
    if max_line_gap is None or max_line_gap < 1:
        max_line_gap = 10
    return min_line_len,max_line_gap

def resize_image(img, max_size=900):
    h,w = img.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(img, (new_w, new_h))