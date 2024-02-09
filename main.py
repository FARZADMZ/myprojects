import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


root = tk.Tk()
root.title("Person Detection App")

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            detect_and_display(img)
        else:
            dimensions_label.config(text="Error: Unable to load the image file.")

def detect_and_display(img):
    height, width, _ = img.shape


    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())


    person_detected = False
    class_id = None
    confidence = None
    box = None

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                box = [x, y, w, h]
                person_detected = True
                break
    if person_detected:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
        heighttt = h * 0.367
        text = f"Width: {w}px\nHeight: {heighttt}cm"
        dimensions_label.config(text=text)
    else:
        dimensions_label.config(text="No person detected")


open_button = tk.Button(root, text="Open Image", command=open_image)
label = tk.Label(root)
dimensions_label = tk.Label(root, text="")


open_button.pack()
label.pack()
dimensions_label.pack()

root.mainloop()
