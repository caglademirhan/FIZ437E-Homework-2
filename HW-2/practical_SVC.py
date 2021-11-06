import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X /255.0

print("fetched !")

X_train, X_test = X[:1000], X[1000:] ;    y_train, y_test = y[:1000], y[1000:]


model_SVC_wr = SVC(C=10, gamma=0.001, kernel="rbf")
model_SVC_wr.fit(X_train, y_train)
y_predict_SVC_wr = model_SVC_wr.predict(X_test)


model_SVC_wor = SVC(C=1000, gamma=0.001, kernel="rbf")
model_SVC_wor.fit(X_train, y_train)
y_predict_SVC_wor = model_SVC_wor.predict(X_test)


print("SVC Model using regularization accuracy: \t", 
        metrics.accuracy_score(y_test, y_predict_SVC_wr), "\n")
print("SVC Model using regularization accuracy: \t",
        metrics.accuracy_score(y_test, y_predict_SVC_wor), "\n")

