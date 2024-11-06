from tkinter import filedialog, simpledialog
import os
from utils import *
from LineMerger import *
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt # type: ignore
from itertools import islice

#pip install git+https://github.com/C-R-Kelly/CircuitSchematicImageInterpreter
from CircuitSchematicImageInterpreter.SPICE import createNetList # type: ignore
from CircuitSchematicImageInterpreter.io import importImage, exportComponent # type: ignore
from CircuitSchematicImageInterpreter.actions import wireScanHough, objectDetection, junctionDetection # type: ignore
from CircuitSchematicImageInterpreter.ocr import OCRComponents # type: ignore
import pytesseract # type: ignore
import os

class LineDetection:
    def __init__(self):
        self.components = []            #components detected in the image from VOC data
        self.detected_lines = None      #original lines detected upon which changes are performed
        self.original_image = None      #original image for display, to simulate real time changes
        self.add_mode = False           #flag for adding lines toggle button
        self.remove_mode = False        #flag for removing lines toggle button
        self.start_point = None         #starting point of the line
        self.original_dimensions = None #original image dimensions
        self.resized_dimensions = None  #resized image dimensions
        self.component_mask = None      #mask for components in the image
        self.show_lines_only = False    #flag for showing only lines
        self.scale = 0.97               #scale for LSD line detection
        self.line_map = {}              #map of lines to lines
        self.line_component_map = {}    #map of lines to components

    def open_image(self,display_image,parent_window):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")],parent=parent_window)
        
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img.copy()
            self.original_dimensions = img.shape[:2]

            self.components.clear()
            self.line_map.clear()
            self.line_component_map.clear()
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
            self.scale = simpledialog.askfloat("Input", "Enter scale of image (0-1) for LSD:", minvalue=0, maxvalue=1, initialvalue=0.97,parent=parent_window)
            self.model_type = simpledialog.askinteger("Input", "Enter model type (0-lsd/1-pretrained):", initialvalue="0",parent=parent_window)
            if self.model_type == 1:
                self.model1(display_image,file_path)
            elif self.model_type == 2:
                self.fclip(file_path,display_image)
            else:
                self.show_image(img,display_image)

    def fclip(self,path,display_image):
        pass
    

    def get_connections(self,display_image):
        class Wire:
            def __init__(self,coordinates,connections,color=None):
                self.coordinates = coordinates 
                self.connections = connections
                self.color = color
                self.drawn = False
            
            def set_color(self,color):
                self.color = color

            def draw(self,img):
                x0,y0,x1,y1 = self.coordinates
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), self.color, 1, cv2.LINE_AA)
                for connection in self.connections:
                    x2,y2,x3,y3 = connection
                    cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), self.color, 1, cv2.LINE_AA)

        empty = np.ones_like(self.original_image) * 255
        colors = [
                (0, 255, 0),       # Green
                (0, 100, 0),       # Dark Green
                (128, 0, 128),     # Purple
                (255, 102, 204),   # Rose
                (255, 0, 0),       # Red
                (0, 0, 255),       # Blue
                (0, 0, 139),       # Dark Blue
                (165, 42, 42),     # Brown
                (128, 0, 128),     # Purple
                (255, 165, 0)      # Orange
            ]
        
        color_index = 0
        wires = {}
        components = {}

        for line, connections in self.line_map.items():
            color = colors[color_index % len(colors)]
            color_index += 1
            wires[line] = Wire(line, connections, color)

        for wire in wires.values():
            wire.draw(empty)

        display_image(resize_image(empty))

    def model1(self,display_image,path):
        #setting tessaract path and data path
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
        os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'

        # Import image
        image = importImage(path)
        # Get Wires
        HorizWires, VertWires = wireScanHough(image)
        # Get Components
        # components = objectDetection(HorizWires, VertWires)

        empty = np.ones(self.original_image.shape[:2], np.uint8) * 255

        for wire in HorizWires:
            y1,y2,x1,x2 = wire.line
            cv2.line(empty, (int(x1), int(y1)), (int(x2), int(y2)), 0, 2, cv2.LINE_AA)

        for wire in VertWires: 
            y1,y2,x1,x2 = wire.line
            cv2.line(empty, (int(x1), int(y1)), (int(x2), int(y2)), 0, 2, cv2.LINE_AA)

        display_image(resize_image(empty))

    def show_image(self,img,display_image):
        lined_image = self.line_detection(img, self.component_mask)
        self.highlight_components(lined_image)
        resized_image = resize_image(lined_image) 
        self.resized_dimensions = resized_image.shape[:2]
        self.merge_lines(display_image,5)
        self.connect_lines(display_image,line_gap=25,plane_gap=5)

    def highlight_components(self, img):
        for rect in self.components:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    def line_detection(self, img, component_mask): 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        empty = np.zeros((img.shape), np.uint8)

        mean_intensity = np.mean(gray)
        if mean_intensity > 127:
            gray = cv2.bitwise_not(gray)

        mask = cv2.bitwise_and(gray, gray, mask=component_mask)
        lsd = cv2.createLineSegmentDetector(scale=self.scale)
        self.detected_lines = lsd.detect(mask)[0]
        self.detected_lines = [l for l in self.detected_lines if not self.is_slanted(*l.flatten())]

        for l in self.detected_lines:
            x0, y0, x1, y1 = l.flatten()
            cv2.line(empty, (int(x0), int(y0)), (int(x1), int(y1)), 255, 2, cv2.LINE_AA)
        
        self.check_lines_near_components()
        img = cv2.addWeighted(img, 1, empty, 1, 0)

        return img

    def check_lines_near_components(self, threshold=3):
        lines_to_remove = []
        for c in self.components:
            xmin, ymin, xmax, ymax = c
            for line in self.detected_lines:
                x0, y0, x1, y1 = line.flatten()
                #line on left
                if (abs(x0 - xmin) <= threshold or abs(x1 - xmin) <= threshold) and self.is_parallel(x0, y0, x1, y1):
                    if (ymin <= y0 <= ymax) or (ymin <= y1 <= ymax):
                        lines_to_remove.append(tuple(line.flatten()))
                        continue

                #line on right
                if (abs(x0 - xmax) <= threshold or abs(x1 - xmax) <= threshold) and self.is_parallel(x0, y0, x1, y1):
                    if (ymin <= y0 <= ymax) or (ymin <= y1 <= ymax):
                        lines_to_remove.append(tuple(line.flatten()))
                        continue

                #line on top
                if (abs(y0 - ymin) <= threshold or abs(y1 - ymin) <= threshold) and self.is_parallel(x0, y0, x1, y1,top=True):
                    if (xmin <= x0 <= xmax) or (xmin <= x1 <= xmax):
                        lines_to_remove.append(tuple(line.flatten()))
                        continue

                #line is below
                if (abs(y0 - ymax) <= threshold or abs(y1 - ymax) <= threshold) and self.is_parallel(x0, y0, x1, y1,True):
                    if (xmin <= x0 <= xmax) or (xmin <= x1 <= xmax):
                        lines_to_remove.append(tuple(line.flatten()))
                        continue

        # Remove lines that are close to any side of the component and parallel
        self.detected_lines = [line for line in self.detected_lines if tuple(line.flatten()) not in lines_to_remove]
    

    #TODO: unoptimized, use tree traversal for faster results ?
    def create_line_map(self,threshold=10):
        def check_connection(l1, l2):
            x0, y0, x1, y1 = l1
            x2, y2, x3, y3 = l2
            
            # Check endpoint connections
            endpoints = [
                (abs(x0-x2) <= threshold and abs(y0-y2) <= threshold),
                (abs(x0-x3) <= threshold and abs(y0-y3) <= threshold),
                (abs(x1-x2) <= threshold and abs(y1-y2) <= threshold),
                (abs(x1-x3) <= threshold and abs(y1-y3) <= threshold)
            ]
            
            if any(endpoints):
                return True
                
            # Check line-to-point connections
            point_to_line = [
                self.is_point_near_line(x2, y2, x0, y0, x1, y1, threshold),
                self.is_point_near_line(x3, y3, x0, y0, x1, y1, threshold),
                self.is_point_near_line(x0, y0, x2, y2, x3, y3, threshold),
                self.is_point_near_line(x1, y1, x2, y2, x3, y3, threshold)
            ]
            
            return any(point_to_line)
        from collections import deque
        self.line_map = {}
        processed_lines = set()
        lines_queue = deque()

        for line in self.detected_lines:
            x0, y0, x1, y1 = line.flatten()
            line = tuple((int(x0),int(y0),int(x1),int(y1)))
            lines_queue.append(line)
            self.line_map[line] = []
        
        while lines_queue:
            current_line = lines_queue.popleft()

            if current_line in processed_lines:
                continue

            for other_line in self.detected_lines:
                if np.any(current_line != other_line):
                    x0,y0,x1,y1 = other_line.flatten()
                    other_line = tuple((int(x0),int(y0),int(x1),int(y1)))
                    if np.all(current_line != other_line):
                        if check_connection(current_line, other_line):
                            self.line_map[current_line].append(other_line)
                            self.line_map[other_line].append(current_line)
            
            processed_lines.add(current_line)

        # Add indirect connections
        for line in self.line_map:
            indirect_connections = set()
            # Check connections of direct connections
            for connected_line in self.line_map[line]:
                for indirect_line in self.line_map[connected_line]:
                    if np.all(line != indirect_line):
                        indirect_connections.add(indirect_line)
            
            # Add indirect connections to line_map
            self.line_map[line].extend(list(indirect_connections))

        return self.line_map
    
    def create_line_component_map(self,threshold=5):
        for c in self.components:
            for line in self.detected_lines:

                if c not in self.line_component_map:
                    self.line_component_map[c] = []

                if self.are_connected(line,c,threshold):
                    x0, y0, x1, y1 = line.flatten()
                    self.line_component_map[c].append(tuple((int(x0),int(y0),int(x1),int(y1))))

        return self.line_component_map
    
    def is_parallel(self, x0, y0, x1, y1,top=False,angle_threshold=5):
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if top:
            # Check if the line is parallel to top/bottom
            return abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold or abs(angle + 180) < angle_threshold
        else:
            # Check if the line is parallel to sides
            return abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold
        
    def is_slanted(self,x0, y0, x1, y1, threshold=20):
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        return not (abs(angle) < threshold or abs(angle - 90) < threshold or abs(angle + 90) < threshold)
    
    ###FILTERING FUNCTIONS###
    def merge_lines(self,display_image,merging_gap):
        new_lines = self.detected_lines.copy()

        bundler = Merger(min_distance=merging_gap)
        new_lines = bundler.process_lines(new_lines)
        new_lines = np.array([line for line in new_lines if not self.is_slanted(line[0][0], line[0][1], line[0][2], line[0][3])])
        self.detected_lines = new_lines
        self.straighten_lines(display_image)

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

    def connect_lines(self,display_image,line_gap,plane_gap=5):
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

        vertical_lines = connect_vertical_lines(vertical_lines,line_gap,plane_gap)
        horizontal_lines = connect_horizontal_lines(horizontal_lines,line_gap,plane_gap)
        vertical_lines,horizontal_lines = connect_perpendicular_lines(vertical_lines,horizontal_lines,line_gap)
        all_lines = np.append(horizontal_lines,vertical_lines,axis=0)
        self.detected_lines = all_lines
        for c in self.components:
            xmin, ymin, xmax, ymax = c
            if abs(xmin-xmax) > abs(ymin-ymax):
                self.detected_lines = np.append(self.detected_lines,[[[xmin,(ymin+ymax)/2,xmin-3,(ymin+ymax)/2]]],axis=0)
        self.create_line_map()
        self.create_line_component_map()
        self.straighten_lines(display_image)  
    
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
                self.start_point = None
                self.show_lines(display_image,self.detected_lines)

    def remove_lines(self, event, display_image, padding=10):
        if self.remove_mode:
            click_point = (event.x, event.y)
            click_scaled = self.scale_to_original(click_point)
            
            for i, line in enumerate(self.detected_lines):
                x1, y1, x2, y2 = line.flatten()
                if self.is_point_near_line(click_scaled[0], click_scaled[1], x1, y1, x2, y2, padding):
                    self.detected_lines = np.delete(self.detected_lines, i, axis=0)
                    break
            self.remove_mode = True
            self.show_lines(display_image,self.detected_lines)

    def is_point_near_line(self,px, py, x1, y1, x2, y2, padding):
        len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if len < 1e-6:
            return False
        #project point onto line
        d = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (len ** 2)
        if d < 0 or d > 1:
            return False
        #find intersection point of line and projection
        ix = x1 + d * (x2 - x1)
        iy = y1 + d * (y2 - y1)
        #check if intersection point is close to the point
        distance = np.sqrt((px - ix) ** 2 + (py - iy) ** 2)
        return distance <= padding
    
    ###DISPLAY FUNCTION###
    def show_lines(self,display_image,new_lines):
        self.detected_lines = new_lines
        if self.show_lines_only:
            # Create a white image
            img = np.ones_like(self.original_image) * 255

            for line in self.detected_lines:
                x1, y1, x2, y2 = line.flatten()
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
                # Draw circles at the beginning and end of each line segment
                cv2.circle(img, (int(x1), int(y1)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(x2), int(y2)), 5, (0, 0, 255), -1)
        else:
            img = np.copy(self.original_image) * 0
            for line in self.detected_lines:
                x1, y1, x2, y2 = line.flatten()
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            img = cv2.addWeighted(self.original_image, 1, img, 1, 0)
        self.highlight_components(img)
        display_image(resize_image(img))

    ###OTHER FUNCTIONS###
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
        self.add_mode = False
        self.remove_mode = False
        self.show_lines_only = False
        self.show_image(self.original_image,display_image)

    def are_connected(self,line,component,threshold=5):
        x0,y0,x1,y1 = line.flatten()
        xmin, ymin, xmax, ymax = component
        return (self.is_point_near_line(x0,y0,xmin,ymin,xmax,ymin,threshold) or self.is_point_near_line(x1, y1, xmin, ymin, xmax, ymin, threshold)
        or self.is_point_near_line(x0,y0,xmin,ymax,xmax,ymax,threshold) or self.is_point_near_line(x1, y1, xmin, ymax, xmax, ymax, threshold)
        or self.is_point_near_line(x0,y0,xmin,ymin,xmin,ymax,threshold) or self.is_point_near_line(x1, y1, xmin, ymin, xmin, ymax, threshold)
        or self.is_point_near_line(x0,y0,xmax,ymin,xmax,ymax,threshold) or self.is_point_near_line(x1, y1, xmax, ymin, xmax, ymax, threshold))
