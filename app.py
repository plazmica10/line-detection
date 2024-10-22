import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog   
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
from utils import *
import math

class LineDetection:
    def __init__(self):
        self.components = []            #components detected in the image from VOC data
        self.original_lines = None      #original lines detected upon which changes are performed
        self.original_image = None      #original image for display, to simulate real time changes
        self.min_line_len = 10          #minimum line length
        self.max_line_gap = 10          #maximum line gap
        self.add_mode = False           #flag for adding lines toggle button
        self.start_point = None         #starting point of the line
        self.original_dimensions = None #original image dimensions
        self.resized_dimensions = None  #resized image dimensions


    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")], parent=self.window)
        
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img.copy()
            self.original_dimensions = img.shape[:2]

            self.components.clear()
            component_mask = np.ones_like(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) * 255
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
                    cv2.rectangle(component_mask, (xmin, ymin), (xmax, ymax), 0, -1)

            self.min_line_len = simpledialog.askinteger("Input", "Enter minimum line length:", minvalue=-1000, maxvalue=1000, initialvalue=10, parent=self.window)
            self.max_line_gap = simpledialog.askinteger("Input", "Enter maximum line gap:", minvalue=-1000, maxvalue=1000, initialvalue=10, parent=self.window)
            self.min_line_len, self.max_line_gap = validate_input(self.min_line_len, self.max_line_gap)

            lined_image = self.line_detection(img, component_mask)
            
            self.highlight_components(lined_image)

            # Resizing because original image overflows the window
            resized_image = resize_image(lined_image) 
            self.resized_dimensions = resized_image.shape[:2]
            self.display_image(resized_image)

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

    def connect_lines(self):
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
        self.update_image(new_lines)
        return new_lines

    def remove_short_lines(self):
        new_lines = np.array([line for line in self.original_lines if np.sqrt((line[0][0] - line[0][2])**2 + (line[0][1] - line[0][3])**2) > self.min_line_len])
        self.update_image(new_lines)

    def update_image(self, new_lines):
        self.original_lines = new_lines
        img = self.original_image.copy()
        line_image = np.copy(self.original_image) * 0

        for line in new_lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        lined_image = cv2.addWeighted(img, 1, line_image, 1, 0)
        self.highlight_components(lined_image)
        lined_image = resize_image(lined_image)
        self.display_image(lined_image)
        
    def straighten_lines(self):
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
        self.update_image(new_lines)

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

    def add_lines(self, event):
        if self.add_mode:
            if self.start_point is None:
                self.start_point = (event.x, event.y)
            else:
                end_point = (event.x, event.y)
                start_scaled = self.scale_to_original(self.start_point)
                end_scaled = self.scale_to_original(end_point)
                new_line = np.array([[start_scaled[0], start_scaled[1], end_scaled[0], end_scaled[1]]])
                self.original_lines = np.append(self.original_lines, [new_line], axis=0)
                self.update_image(self.original_lines)
                self.start_point = None
            
    def display_image(self, img):
        # Converting to format for tkinter
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.grid_propagate(False)
        self.image_frame.config(width=img.shape[1], height=img.shape[0])
        self.image_label = tk.Label(self.image_frame, image=img_tk, bd=2, relief='solid')
        self.image_label.img = img_tk
        self.image_label.grid(column=0, row=0)
        self.image_label.bind("<Button-1>", self.add_lines)

    def display_window(self):
        self.window = tk.Tk()   
        self.window.wm_attributes('-topmost', 1)
        self.window.title("Line Detection")
        self.window.geometry('1280x720')

        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.window)
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.mainframe.columnconfigure(0, weight=0)
        self.mainframe.columnconfigure(1, weight=1)
        self.mainframe.rowconfigure(1, weight=0)

        sidebar = ttk.Frame(self.mainframe, padding='3 3 12 12')
        sidebar.grid(column=0, row=0, rowspan=2, sticky=(tk.N, tk.W, tk.E, tk.S))

        ttk.Button(sidebar, text="Connect Lines", command=self.connect_lines).grid(column=0, row=0, sticky=(tk.W))
        ttk.Button(sidebar, text="Straighten Lines", command=self.straighten_lines).grid(column=0, row=1, sticky=(tk.W))
        ttk.Button(sidebar, text="Remove Short Lines", command=self.remove_short_lines).grid(column=0, row=2, sticky=(tk.W))
        ttk.Button(sidebar, text="Toggle Add-Line Mode", command=self.toggle_mode).grid(column=0, row=3, sticky=(tk.W))

        self.image_frame = ttk.Frame(self.mainframe)
        self.image_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        ttk.Button(self.mainframe, text="Open Image", command=self.open_image).grid(column=2, row=1, sticky=(tk.W))

        self.window.mainloop()

class Merger:     
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing grops, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)
   

if __name__ == '__main__':
    app = LineDetection()
    app.display_window()