import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('handwrittenRecognition.keras')

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

predictions = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

report = classification_report(y_test, predicted_classes, digits=4)
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(y_test, predicted_classes)
print("\nConfusion Matrix:\n", conf_matrix)
print("\n")

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
individual_losses = loss_fn(y_test, predictions).numpy()

results = {digit: {"loss": [], "correct": 0, "total": 0} for digit in range(10)}

for i in range(len(y_test)):
    true_label = y_test[i]
    predicted_label = predicted_classes[i]
    
    loss_pom = loss_fn(np.array([true_label]), np.array([predictions[i]])).numpy()
    
    results[true_label]["loss"].append(loss_pom)
    results[true_label]["total"] += 1
    if predicted_label == true_label:
        results[true_label]["correct"] += 1

for digit in range(10):
    avg_loss = np.mean(results[digit]["loss"]) if results[digit]["loss"] else 0
    accuracy_pom = results[digit]["correct"] / results[digit]["total"] if results[digit]["total"] > 0 else 0
    print(f"Cyfra {digit}: Loss = {avg_loss:.4f}, Accuracy = {accuracy_pom:.4f}")

print("Average Loss :", loss)
print("Average Accuracy :", accuracy)

