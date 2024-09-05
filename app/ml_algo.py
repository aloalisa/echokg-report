import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_answer_for_test(xTest):
    dataset = np.loadtxt(fname="./final_dataset.csv", delimiter=",")
    col_count = 10
    X = dataset[:, 0:col_count]
    y = dataset[:, col_count]

    X_train = X
    y_train = y
    X_test = xTest

    results = [0, 0, 0]

    # Decision tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1
    print(predicted)

    # Logistic regression
    model = LogisticRegression(C=1.0)
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1
    print(predicted)

    # Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1
    print(predicted)

    # K blijaywi sosed
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1
    print(predicted)

    #  Perceptron
    model = Perceptron()
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1
    print(predicted)

    # Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1   
    print(predicted)
    print(model.predict_proba(X_test))

    # Metod opornix vektor
    model = SVC()
    model.fit(X_train, y_train)
    predicted = int(model.predict(X_test)[0])
    results[predicted] += 1
    print(predicted)

    sm = sum(results)
    for i in range(len(results)):
        results[i] = round(results[i] * 100 / sm, 1)

    return results
