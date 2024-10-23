from tkinter import filedialog, simpledialog
import os
from utils import *
from LineMerger import *
import xml.etree.ElementTree as ET

class LineDetection:
    def __init__(self):
        self.components = []            #components detected in the image from VOC data
        self.original_lines = None      #original lines detected upon which changes are performed
        self.original_image = None      #original image for display, to simulate real time changes
        self.min_line_len = None          #minimum line length
        self.max_line_gap = None          #maximum line gap
        self.add_mode = False           #flag for adding lines toggle button
        self.start_point = None         #starting point of the line
        self.original_dimensions = None #original image dimensions
        self.resized_dimensions = None  #resized image dimensions
        self.component_mask = None      #mask for components in the image
    def open_image(self,display_image,parent_window):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")],parent=parent_window)
        
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img.copy()
            self.original_dimensions = img.shape[:2]

            self.components.clear()
            self.component_mask = np.ones_like(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) * 255
            # Load VOC data
            voc_file_path = file_path.replace('.jpg', '.xml').replace('.png', '.xml')
            if os.path.exists(voc_file_path):
                tree = ET.parse(voc_file_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    self.components.append((xmin, ymin, xmax, ymax))
                    cv2.rectangle(self.component_mask, (xmin, ymin), (xmax, ymax), 0, -1)

            self.input_dialog(parent_window)
            
            self.show_image(img,display_image)

    def show_image(self,img,display_image):
        lined_image = self.line_detection(img, self.component_mask)
        
        self.highlight_components(lined_image)

        # Resizing because original image overflows the window
        resized_image = resize_image(lined_image) 
        self.resized_dimensions = resized_image.shape[:2]
        display_image(resized_image)


    def input_dialog(self,parent_window):
        self.min_line_len = simpledialog.askinteger("Input", "Enter minimum line length:", minvalue=-1000, maxvalue=1000, initialvalue=10,parent=parent_window)
        self.max_line_gap = simpledialog.askinteger("Input", "Enter maximum line gap:", minvalue=-1000, maxvalue=1000, initialvalue=10,parent=parent_window)
        self.min_line_len, self.max_line_gap = validate_input(self.min_line_len, self.max_line_gap)


    def highlight_components(self, img):
        for rect in self.components:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    def line_detection(self, img, component_mask): 

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #amplifying lines in the image
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i, j] > 40:
                    gray[i, j] = 255
        # LSD method
        empty = np.zeros((img.shape), np.uint8)
        mask = cv2.bitwise_and(gray, gray, mask=component_mask)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(mask)[0]
        self.original_lines = lines

        for l in lines:
            x0, y0, x1, y1 = l.flatten()
            cv2.line(empty, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2, cv2.LINE_AA)

        img = cv2.addWeighted(img, 1, empty, 1, 0)

        return img

    def connect_lines(self,display_image):
        def are_orthogonal(line1, line2):
            x1, y1, x2, y2 = line1.flatten()
            x3, y3, x4, y4 = line2.flatten()
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x4 - x3, y4 - y3
            dot_product = np.dot([dx1, dy1], [dx2, dy2])
            return abs(dot_product) < 1
        
        def extend_line(line, length=5):
            x1, y1, x2, y2 = line.flatten()
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / norm, dy / norm
            new_x2, new_y2 = x2 + dx * length, y2 + dy * length
            return np.array([[x1, y1, new_x2, new_y2]])
        
        max_extension_length = self.max_line_gap
        new_lines = self.original_lines.copy()
        for i in range(len(self.original_lines)):
            for j in range(i + 1, len(self.original_lines)):
                line1 = self.original_lines[i]
                line2 = self.original_lines[j]
                if are_orthogonal(line1, line2):
                    extended_line1 = extend_line(line1, max_extension_length)
                    extended_line2 = extend_line(line2, max_extension_length)
                    new_lines = np.append(new_lines, [extended_line1, extended_line2], axis=0)

        bundler = Merger()
        new_lines = bundler.process_lines(new_lines)
        print(len(new_lines))
        self.update_image(new_lines,display_image)
        return new_lines

    def remove_short_lines(self,display_image):
        new_lines = np.array([line for line in self.original_lines if np.sqrt((line[0][0] - line[0][2])**2 + (line[0][1] - line[0][3])**2) > self.min_line_len])
        self.update_image(new_lines,display_image)

    def update_image(self, new_lines, display_image):
        self.original_lines = new_lines
        img = self.original_image.copy()
        line_image = np.copy(self.original_image) * 0

        for line in new_lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        lined_image = cv2.addWeighted(img, 1, line_image, 1, 0)
        self.highlight_components(lined_image)
        lined_image = resize_image(lined_image)
        display_image(lined_image)
        
    def straighten_lines(self,display_image):
        new_lines = []
        for line in self.original_lines:
            x1, y1, x2, y2 = line.flatten()
            if(abs(x1 - x2) > self.min_line_len or abs(y1 - y2) > self.min_line_len):
                if abs(x1 - x2) > abs(y1 - y2):
                    # Make y-coordinates the same
                    y1 = y2 = (y1 + y2) // 2
                else:
                    # Make x-coordinates the same
                    x1 = x2 = (x1 + x2) // 2
            new_lines.append([[x1, y1, x2, y2]])
        new_lines = np.array(new_lines)
        self.update_image(new_lines,display_image)

    def scale_to_original(self, point):
        """Scale a point from the resized image to the original image dimensions."""
        resized_h, resized_w = self.resized_dimensions
        original_h, original_w = self.original_dimensions
        x, y = point
        
        # Calculate scaling factors
        scale_x = original_w / resized_w
        scale_y = original_h / resized_h
        
        # Apply scaling factors to the input point
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        
        return (scaled_x, scaled_y)
    
    def toggle_mode(self):
        self.add_mode = not self.add_mode

    def add_lines(self, event, display_image):
        if self.add_mode:
            if self.start_point is None:
                self.start_point = (event.x, event.y)
            else:
                end_point = (event.x, event.y)
                start_scaled = self.scale_to_original(self.start_point)
                end_scaled = self.scale_to_original(end_point)
                new_line = np.array([[start_scaled[0], start_scaled[1], end_scaled[0], end_scaled[1]]])
                self.original_lines = np.append(self.original_lines, [new_line], axis=0)
                self.update_image(self.original_lines,display_image)
                self.start_point = None
    def update_length(self, min_line_len):
        self.min_line_len = min_line_len
        print(self.min_line_len)
        
    def update_gap(self, max_line_gap):
        self.max_line_gap = max_line_gap
        print(self.max_line_gap)

    def get_length(self):
        return self.min_line_len
    
    def get_gap(self):
        return self.max_line_gap
    
    def refresh(self,display_image):
        self.show_image(self.original_image,display_image)

            