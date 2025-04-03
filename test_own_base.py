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
        try:
            true_label = int(file_name[0])
            true_labels.append(true_label)
            print(f"File: {file_name}, True Label: {true_label}")

            file_path = os.path.join(digits_folder, file_name)
            img = Image.open(file_path).convert("L")  
            img = img.resize((28, 28))  
            img = np.array(img)
            img = cv2.bitwise_not(img)  
            img = img / 255.0  
            img = img.reshape(1, 28, 28, 1) 

            predictions = model.predict(img, verbose=0)
            predicted_label = np.argmax(predictions)
            predicted_labels.append(predicted_label)
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {file_name}: {e}")

report = classification_report(true_labels, predicted_labels, digits=4)
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:\n", conf_matrix)

results = {digit: {"correct": 0, "total": 0} for digit in range(10)}

for true_label, predicted_label in zip(true_labels, predicted_labels):
    results[true_label]["total"] += 1
    if true_label == predicted_label:
        results[true_label]["correct"] += 1

for digit in range(10):
    total = results[digit]["total"]
    accuracy = results[digit]["correct"] / total if total > 0 else 0
    print(f"Cyfra {digit}: Accuracy = {accuracy:.4f}")

for file_name, true_label, predicted_label in zip(os.listdir(digits_folder), true_labels, predicted_labels):
    print(f"File: {file_name}, True Label: {true_label}, Predicted Label: {predicted_label}")

