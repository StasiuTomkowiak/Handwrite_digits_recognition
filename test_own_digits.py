import tkinter as tk
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw

model = tf.keras.models.load_model('handwrittenRecognition.keras')

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rysuj cyfrę i rozpoznaj ją!")

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.btn_recognize = tk.Button(root, text="Rozpoznaj", command=self.recognize_digit)
        self.btn_recognize.pack()

        self.btn_clear = tk.Button(root, text="Wyczyść", command=self.clear_canvas)
        self.btn_clear.pack()

        self.image = Image.new("L", (280, 280), 255)  
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")  

    def recognize_digit(self):

        img = self.image.resize((28, 28))  
        img = np.array(img)
        img = cv2.bitwise_not(img) 
        img = img / 255.0  
        img = img.reshape(1, 28, 28, 1)  

        predictions = model.predict(img)
        predicted_digit = np.argmax(predictions)

        result_label.config(text=f"Rozpoznana cyfra: {predicted_digit}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

root = tk.Tk()
app = DigitRecognizerApp(root)

result_label = tk.Label(root, text="Narysuj cyfrę i kliknij 'Rozpoznaj'", font=("Arial", 14))
result_label.pack()

root.mainloop()
