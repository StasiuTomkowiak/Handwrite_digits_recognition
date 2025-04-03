import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(x_train, y_train)

predicted_classes = forest.predict(x_test)

accuracy = accuracy_score(y_test, predicted_classes)
print(f'Accuracy: {accuracy:.4f}')


report = classification_report(y_test, predicted_classes, digits=4)
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(y_test, predicted_classes)
print("\nConfusion Matrix:\n", conf_matrix)
print("\n")

