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
        self.image_label.bind("<Button-1>", lambda event: self.line_detection.add_lines(event, self.display_image))

    def display_window(self):
        self.window = tk.Tk()   
        self.show_lines_var = tk.BooleanVar()
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
        sidebar.rowconfigure(6, weight=1)  # Add weight to the row before the button to push it to the bottom

        ttk.Button(sidebar, text="Merge Lines", command=lambda: self.line_detection.merge_lines(self.display_image)).grid(column=0, row=0, sticky=(tk.W))
        ttk.Button(sidebar, text="Straighten Lines", command=lambda: self.line_detection.straighten_lines(self.display_image)).grid(column=0, row=1, sticky=(tk.W))
        self.min_line_len = tk.IntVar()
        ttk.Entry(sidebar,textvariable=self.min_line_len,width=10).grid(column=1, row=2, sticky=(tk.W))
        ttk.Button(sidebar, text="Remove Short Lines", command=lambda: self.line_detection.remove_short_lines(self.display_image,self.min_line_len.get())).grid(column=0, row=2, sticky=(tk.W))
        ttk.Button(sidebar, text="Toggle Add-Line Mode", command=self.line_detection.toggle_mode).grid(column=0, row=3, sticky=(tk.W))

        ttk.Checkbutton(
            sidebar, 
            text="Show only lines", 
            variable=self.show_lines_var, 
            command=self.toggle_show_lines
        ).grid(column=0, row=4, sticky=(tk.W))

        ttk.Button(sidebar, text="Refresh", command=lambda: self.line_detection.refresh(self.display_image)).grid(column=0, row=6, sticky=(tk.W,tk.S))

        self.image_frame = ttk.Frame(self.mainframe)
        self.image_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.image_frame.config(width=1100, height=600)  # Reserve 500px for the image_frame

        ttk.Button(self.mainframe, text="Open Image", command=lambda: self.open_image()).grid(column=1, row=1, sticky=(tk.E))

        self.window.mainloop()
    
    def open_image(self):
        self.line_detection.open_image(self.display_image,self.window)

    def toggle_show_lines(self):
        if self.show_lines_var.get():
            self.line_detection.show_lines_var = True
            self.line_detection.display_lines_and_components(self.display_image)
        else:
            self.line_detection.show_lines_var = False
            self.line_detection.display_lined_image(self.display_image)


if __name__ == '__main__':
    line_detector = LineDetection()
    gui = GUI(line_detector)
    gui.display_window()