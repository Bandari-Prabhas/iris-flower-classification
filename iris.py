import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)

model = GaussianNB()
# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Evaluate the model's accuracy
accuracy = (y_pred == y_test).mean()
print("Accuracy:",accuracy)

class_accuracy=np.mean(y_pred==y_test,axis=0)
class_labels=['Setosa','Versicolor','Virginica']
accuracy_values=class_accuracy.tolist()
plt.bar(class_labels,accuracy_values)
plt.xlabel("Class Lables")
plt.ylabel("Accuracy")
plt.title("Navie Bais Classifier using -Iris Data set")
plt.show()