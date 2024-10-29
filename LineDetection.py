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
        self.remove_mode = False
        self.start_point = None         #starting point of the line
        self.original_dimensions = None #original image dimensions
        self.resized_dimensions = None  #resized image dimensions
        self.component_mask = None      #mask for components in the image
        self.show_lines_only = False    #flag for showing only lines
        self.scale = 0.97               #scale for LSD line detection

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
            self.scale = simpledialog.askfloat("Input", "Enter scale of image (0-1) for LSD:", minvalue=0, maxvalue=1, initialvalue=0.97,parent=parent_window)
            self.show_image(img,display_image)

    def show_image(self,img,display_image):
        lined_image = self.line_detection(img, self.component_mask)
        self.highlight_components(lined_image)
        resized_image = resize_image(lined_image) 
        self.resized_dimensions = resized_image.shape[:2]
        display_image(resized_image)

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
        img = cv2.addWeighted(img, 1, empty, 1, 0)
        return img
    
    def is_slanted(self,x0, y0, x1, y1, threshold=20):
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        return not (abs(angle) < threshold or abs(angle - 90) < threshold or abs(angle + 90) < threshold)
    
    ###FILTERING FUNCTIONS###
    def merge_lines(self,display_image,merging_gap):
        new_lines = self.detected_lines.copy()

        bundler = Merger(min_distance=merging_gap)
        new_lines = bundler.process_lines(new_lines)
        new_lines = np.array([line for line in new_lines if not self.is_slanted(line[0][0], line[0][1], line[0][2], line[0][3])])
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
        self.show_lines(display_image,all_lines)
                    
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

            def is_point_near_line(px, py, x1, y1, x2, y2, padding):
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
            
            for i, line in enumerate(self.detected_lines):
                x1, y1, x2, y2 = line.flatten()
                if is_point_near_line(click_scaled[0], click_scaled[1], x1, y1, x2, y2, padding):
                    self.detected_lines = np.delete(self.detected_lines, i, axis=0)
                    break
            self.remove_mode = True
            self.show_lines(display_image,self.detected_lines)
        
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