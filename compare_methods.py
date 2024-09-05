import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def calculate_data(expected, predicted):
    length = len(expected)
    count = sum(1 for exp, pred in zip(expected, predicted) if exp == pred)
    print("Total =", length)
    print("Equals =", count, f"{count / length * 100:.2f}%")
    print("Not Equals =", length - count, f"{(length - count) / length * 100:.2f}%")
    print("------------------------------------------------------------------------")

# Load data
dataset = np.loadtxt(fname='final_dataset.csv', delimiter=',')
X = dataset[:, :10]
y = dataset[:, 10]
np.random.seed(0)
indices = np.random.permutation(len(X))
k = int(-len(X) * 0.15)
X_train, y_train = X[indices[:k]], y[indices[:k]]
X_test, y_test = X[indices[k:]], y[indices[k:]]

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=2),
    "Logistic Regression": LogisticRegression(C=0.01, penalty='l2'),
    "Naive Bayes": GaussianNB(),
    "K Nearest Neighbors": KNeighborsClassifier(metric='manhattan', n_neighbors=9, weights='distance'),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Perceptron": Perceptron()
}

# Training and testing
for name, model in models.items():
    model.fit(X_train, y_train)
    print(model)
    expected = y_test
    predicted = model.predict(X_test)
    calculate_data(expected, predicted)
