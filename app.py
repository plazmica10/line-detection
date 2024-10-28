import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk   
from LineDetection import *

class GUI:
    def __init__(self,line_detector):
        self.line_detection = line_detector

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
        self.update_mode()

    def display_window(self):
        self.window = tk.Tk()   
        self.window.wm_attributes('-topmost', 1)
        self.window.title("Line Detection")
        # self.window.geometry('1280x720')x

        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        self.mainframe = ttk.Frame(self.window)
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.mainframe.columnconfigure(0, weight=0)
        self.mainframe.columnconfigure(1, weight=1)
        self.mainframe.rowconfigure(1, weight=0)

        sidebar = ttk.Frame(self.mainframe, padding='3 3 12 12')
        sidebar.grid(column=0, row=0, rowspan=2, sticky=(tk.N, tk.W, tk.E, tk.S))
        sidebar.rowconfigure(7, weight=1)  # Add weight to the row before the button to push it to the bottom

        self.image_frame = ttk.Frame(self.mainframe)
        self.image_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.image_frame.config(width=1100, height=600)  # Reserve 500px for the image_frame
        
        self.merging_gap = tk.IntVar(value=5)
        ttk.Entry(sidebar,textvariable=self.merging_gap,width=10).grid(column=1, row=0, sticky=(tk.W))
        ttk.Button(
            sidebar,
            text="Merge Lines",
            command=lambda: self.line_detection.merge_lines(self.display_image,self.merging_gap.get())
        ).grid(column=0, row=0, sticky=(tk.W))
        
        ttk.Button(
            sidebar,
            text="Straighten Lines",
            command=lambda: self.line_detection.straighten_lines(self.display_image)
        ).grid(column=0, row=1, sticky=(tk.W))
        
        self.min_line_len = tk.IntVar(value=2)
        ttk.Entry(sidebar,textvariable=self.min_line_len,width=10).grid(column=1, row=2, sticky=(tk.W))
        ttk.Button(
            sidebar,
            text="Remove Short Lines",
            command=lambda: self.line_detection.remove_short_lines(self.display_image,self.min_line_len.get())
        ).grid(column=0, row=2, sticky=(tk.W))
        
        self.connecting_treshold = tk.IntVar(value=25)
        ttk.Entry(sidebar,textvariable=self.connecting_treshold,width=10).grid(column=1, row=3, sticky=(tk.W))
        ttk.Button(
            sidebar,
            text="Connect Lines",
            command=lambda: self.line_detection.connect_lines(self.display_image,self.connecting_treshold.get())
        ).grid(column=0, row=3, sticky=(tk.W))

        self.add_var = tk.BooleanVar()
        ttk.Checkbutton(
            sidebar,
            text="Add Lines",
            variable=self.add_var,
            command=self.add_mode
        ).grid(column=0, row=4, sticky=(tk.W))

        self.remove_var = tk.BooleanVar()
        ttk.Checkbutton(
            sidebar,
            text="Remove Lines",
            variable=self.remove_var,
            command=self.remove_mode 
        ).grid(column=0, row=5, sticky=(tk.W))

        self.show_lines_only = tk.BooleanVar()
        ttk.Checkbutton(
            sidebar, 
            text="Show only lines", 
            variable=self.show_lines_only, 
            command=self.toggle_show_lines
        ).grid(column=0, row=6, sticky=(tk.W))

        self.lsd_scale = tk.StringVar(value=self.line_detection.scale)
        ttk.Entry(sidebar,textvariable=self.lsd_scale,width=10).grid(column=1, row=7, sticky=(tk.S))
        ttk.Button(
            sidebar,
            text="Refresh",
            command=self.refresh_filters
        ).grid(column=0, row=7, sticky=(tk.W,tk.S))


        ttk.Button(
            self.mainframe,
            text="Open Image",
            command=lambda: self.open_image()
        ).grid(column=1, row=1, sticky=(tk.E))

        self.window.mainloop()
    
    def open_image(self):
        self.line_detection.open_image(self.display_image,self.window)

    def toggle_show_lines(self):
        if self.show_lines_only.get():
            self.line_detection.show_lines_only = True
            self.line_detection.display_lines_and_components(self.display_image)
        else:
            self.line_detection.show_lines_only = False
            self.line_detection.display_lined_image(self.display_image)

    def refresh_filters(self):
        self.show_lines_only.set(False)
        self.add_var.set(False)
        self.remove_var.set(False)
        self.line_detection.scale = float(self.lsd_scale.get())
        self.line_detection.refresh(self.display_image)

    def update_mode(self):
        if self.line_detection.add_mode:
            self.image_label.bind("<Button-1>", lambda event: self.line_detection.add_lines(event, self.display_image))
        elif self.line_detection.remove_mode:
            self.image_label.bind("<Button-1>", lambda event: self.line_detection.remove_lines(event, self.display_image))

    def add_mode(self):
        self.line_detection.add_mode = not self.line_detection.add_mode
        self.line_detection.remove_mode = False
        self.remove_var.set(False)
        self.update_mode()

    def remove_mode(self):
        self.line_detection.remove_mode = not self.line_detection.remove_mode
        self.line_detection.add_mode = False
        self.add_var.set(False)
        self.update_mode()

if __name__ == '__main__':
    line_detector = LineDetection()
    gui = GUI(line_detector)
    gui.display_window()