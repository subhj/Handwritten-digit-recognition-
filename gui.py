import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
from keras.models import load_model



model = load_model("annmodelav.h5")

# gui creation
class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root 
        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.label = tk.Label(self.root, text="Draw a digit", font=("Helvetica", 18))
        self.label.pack()

        self.recognize_button = tk.Button(self.root, text="Recognize", command=self.recognize_digit)
        self.recognize_button.pack()

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height))
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def recognize_digit(self):
        image_resized = self.image.resize((28, 28))
        image_data = np.array(image_resized)
        image_data = image_data.reshape(1, 28, 28, 1) / 255.0

        result = model.predict(image_data)
        predicted_digit = np.argmax(result)
        messagebox.showinfo("Recognition Result", f"The digit is: {predicted_digit}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height))
        self.draw = ImageDraw.Draw(self.image)



root = tk.Tk()
root.title("Handwritten Digit Recognition")


app = DigitRecognizerGUI(root)


root.mainloop()
