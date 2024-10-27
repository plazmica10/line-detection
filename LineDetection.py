from tkinter import filedialog, simpledialog
import os
from utils import *
from LineMerger import *
import xml.etree.ElementTree as ET

class LineDetection:
    def __init__(self):
        self.components = []            #components detected in the image from VOC data
        self.detected_lines = None      #original lines detected upon which changes are performed
        self.original_image = None      #original image for display, to simulate real time changes
        self.add_mode = False           #flag for adding lines toggle button
        self.start_point = None         #starting point of the line
        self.original_dimensions = None #original image dimensions
        self.resized_dimensions = None  #resized image dimensions
        self.component_mask = None      #mask for components in the image
        self.show_lines_only = False     #flag for showing only lines

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

            self.show_image(img,display_image)

    def show_image(self,img,display_image):
        lined_image = self.line_detection(img, self.component_mask)
        self.highlight_components(lined_image)
        # Resizing because original image overflows the window
        resized_image = resize_image(lined_image) 
        self.resized_dimensions = resized_image.shape[:2]
        display_image(resized_image)

    def highlight_components(self, img):
        for rect in self.components:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    def line_detection(self, img, component_mask): 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        empty = np.zeros((img.shape), np.uint8)
        mask = cv2.bitwise_and(gray, gray, mask=component_mask)
        lsd = cv2.createLineSegmentDetector(scale=0.95)
        self.detected_lines = lsd.detect(mask)[0]

        for l in self.detected_lines:
            x0, y0, x1, y1 = l.flatten()
            if not self.is_diagonal_line(x0, y0, x1, y1):
                cv2.line(empty, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2, cv2.LINE_AA)
        img = cv2.addWeighted(img, 1, empty, 1, 0)

        return img
    
    def is_diagonal_line(self,x0, y0, x1, y1, threshold=30):
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        return not (abs(angle) < threshold or abs(angle - 90) < threshold or abs(angle + 90) < threshold)
    
    ###FILTERING FUNCTIONS###
    def merge_lines(self,display_image,merging_gap):
        new_lines = self.detected_lines.copy()

        bundler = Merger(min_distance=merging_gap)
        new_lines = bundler.process_lines(new_lines)
        new_lines = np.array([line for line in new_lines if not self.is_diagonal_line(line[0][0], line[0][1], line[0][2], line[0][3])])
        self.show_lines(display_image,new_lines)

    def remove_short_lines(self,display_image,line_lengh):
        new_lines = np.array([line for line in self.detected_lines if np.sqrt((line[0][0] - line[0][2])**2 + (line[0][1] - line[0][3])**2) > line_lengh])
        self.show_lines(display_image,new_lines)
        
    def straighten_lines(self,display_image):
        new_lines = []
        for line in self.detected_lines:
            x1, y1, x2, y2 = line.flatten()
            if abs(x1 - x2) > abs(y1 - y2):
                # Make y-coordinates the same
                y1 = y2 = (y1 + y2) // 2
            else:
                # Make x-coordinates the same
                x1 = x2 = (x1 + x2) // 2
            new_lines.append([[x1, y1, x2, y2]])
        new_lines = np.array(new_lines)
        self.show_lines(display_image,new_lines)

    def connect_lines(self,display_image,line_gap):
        horizontal_lines = []
        vertical_lines = []

        for line in self.detected_lines:
            x0,y0,x1,y1 = line.flatten()
            if(abs(x0 - x1) > abs(y0-y1)):
                horizontal_lines.append(line)
            else:
                vertical_lines.append(line)
        horizontal_lines = np.array(horizontal_lines)
        vertical_lines = np.array(vertical_lines)

        vertical_lines = self.connect_vertical_lines(vertical_lines,line_gap)
        horizontal_lines = self.connect_horizontal_lines(horizontal_lines,line_gap)
        vertical_lines,horizontal_lines = self.connect_perpendicular_lines(vertical_lines,horizontal_lines,line_gap)

        all_lines = np.append(horizontal_lines,vertical_lines,axis=0)
        self.show_lines(display_image,all_lines)
    
    def connect_vertical_lines(self,vertical_lines,line_gap):
        new_vertical_lines = []
        for i in range(len(vertical_lines)):
            x0,y0,x1,y1 = vertical_lines[i].flatten()
            if x0 == 0:
                continue
            for j in range(i+1,len(vertical_lines)):
                x2,y2,x3,y3 = vertical_lines[j].flatten()
                #checking if they are the same line and close enough, also if they are long enough (needs to be removed)
                if abs(x0-x2) < line_gap and (abs(y1-y2) < line_gap or abs(y0-y3) < line_gap) and abs(y0-y1) > line_gap and abs(y2-y3) > line_gap:
                    merged_line = [x0,min(y0,y2),x2,max(y1,y3)]
                    vertical_lines[i] = merged_line
                    vertical_lines[j] = [[0,0,0,0]]
            new_vertical_lines.append(vertical_lines[i])
        new_vertical_lines = np.array([line for line in new_vertical_lines if line[0][0] != 0])
        return new_vertical_lines
    
    def connect_horizontal_lines(slef,horizontal_lines,line_gap):
        new_horizontal_lines = []
        for i in range(len(horizontal_lines)):
            x0,y0,x1,y1 = horizontal_lines[i].flatten()
            if y0 == 0:
                continue
            for j in range(i+1,len(horizontal_lines)):
                x2,y2,x3,y3 = horizontal_lines[j].flatten()
                #checking if they are the same line and close enough, also if they are long enough (needs to be removed)
                if abs(y0-y2) < line_gap and (abs(x1-x2) < line_gap or abs(x0-x3) < line_gap) and abs(x0-x1) > line_gap and abs(x2-x3) > line_gap:
                    merged_line = [min(x0,x2),y0,max(x1,x3),y2]
                    horizontal_lines[i] = merged_line
                    horizontal_lines[j] = [[0,0,0,0]]
            new_horizontal_lines.append(horizontal_lines[i])
        new_horizontal_lines = np.array([line for line in new_horizontal_lines if line[0][1] != 0])
        return new_horizontal_lines

    def connect_perpendicular_lines(self,vertical_lines,horizontal_lines,line_gap):
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
                    
    def add_lines(self, event, display_image):
        if self.add_mode:
            if self.start_point is None:
                self.start_point = (event.x, event.y)
            else:
                end_point = (event.x, event.y)
                start_scaled = self.scale_to_original(self.start_point)
                end_scaled = self.scale_to_original(end_point)
                new_line = np.array([[start_scaled[0], start_scaled[1], end_scaled[0], end_scaled[1]]])
                self.detected_lines = np.append(self.detected_lines, [new_line], axis=0)
                if self.show_lines_only:
                    self.display_lines_and_components(display_image)
                else:
                    self.display_lined_image(display_image)
                self.start_point = None

    ###DISPPLAY FUNCTIONS###
    def show_lines(self,display_image,new_lines):
        self.detected_lines = new_lines
        if self.show_lines_only:
            self.display_lines_and_components(display_image)
        else:
            self.display_lined_image(display_image)
            
    def display_lined_image(self, display_image):
        img = self.original_image.copy()
        line_image = np.copy(self.original_image) * 0

        for line in self.detected_lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        lined_image = cv2.addWeighted(img, 1, line_image, 1, 0)
        self.highlight_components(lined_image)
        lined_image = resize_image(lined_image)
        display_image(lined_image)
        
    def display_lines_and_components(self, display_image):
        # Create a white image
        white_image = np.ones_like(self.original_image) * 255
        # Draw each line segment
        for line in self.detected_lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(white_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
            # Draw circles at the beginning and end of each line segment
            cv2.circle(white_image, (int(x1), int(y1)), 5, (255, 0, 0), -1)
            cv2.circle(white_image, (int(x2), int(y2)), 5, (0, 0, 255), -1)

        # Highlight components
        self.highlight_components(white_image)
        # Display the resulting image
        display_image(resize_image(white_image))

    ###UTILITY FUNCTIONS###
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
    
    def refresh(self,display_image):
        self.show_image(self.original_image,display_image)
        self.show_lines_only = False

    def toggle_mode(self):
        self.add_mode = not self.add_mode
