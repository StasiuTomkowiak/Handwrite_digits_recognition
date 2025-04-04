import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import cv2

digits_folder = "digits"

model = tf.keras.models.load_model('handwrittenRecognition.keras')

true_labels = []
predicted_labels = []

for file_name in os.listdir(digits_folder):
    if file_name.endswith(".jpg"):  
            true_label = int(file_name[0])
            true_labels.append(true_label)

            file_path = os.path.join(digits_folder, file_name)
            img = Image.open(file_path).convert("L")  
            img = np.array(img)
            img = cv2.bitwise_not(img)  
            img = img / 255.0  
            img = img.reshape(28, 28, 1)
            predictions = model.predict(img, verbose=0)
            predicted_label = np.argmax(predictions)
            predicted_labels.append(predicted_label)

report = classification_report(true_labels, predicted_labels, digits=4)
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:\n", conf_matrix)


