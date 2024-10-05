from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
print(data.feature_names)
print(data.target_names)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

# We check on test data to ensure training was good
print(clf.score(x_test, y_test))

# We could check if certain tumor is malignant or benign
# Example of data
synthetic_tumor = [14.0, 20.5, 90.0, 600.0, 0.1, 0.09, 0.07, 0.05, 0.18, 0.06, 0.3, 1.5, 2.1, 30.0, 0.007, 0.025, 0.04, 0.012, 0.02, 0.003, 16.0, 25.0, 110.0, 900.0, 0.14, 0.18, 0.22, 0.09, 0.3, 0.08]
prediction = clf.predict([synthetic_tumor])
print(f"Prediction for the synthetic tumor: {'Malignant' if prediction[0] == 1 else 'Benign'}")

