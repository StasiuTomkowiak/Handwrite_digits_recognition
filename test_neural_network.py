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


