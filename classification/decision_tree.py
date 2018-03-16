import numpy as np
from sklearn import tree
from test_data import evaluate

# load data
X_names = [ "handL", "handR", "Pet", "NoPet", "experience", "age", "distance", 
            "angle", "Male","Female"]

def preprocess_features(X):
    X -= X.mean(axis=0)
    dev = X.std(axis=0)
    dev[dev==0] = 1.0
    X /= dev

    return X

def preprocess_labels(Y):
    threshold = 0.8
    Y[Y>threshold] = 1.0
    Y[Y<=threshold] = 0.0

    return Y

X = np.load("proximity_train_features.npy")
Y = np.load("proximity_train_labels.npy")
X = preprocess_features(X)
Y = preprocess_labels(Y)

X_test = np.load("proximity_test_features.npy")
Y_test = np.load("proximity_test_labels.npy")
X_test = preprocess_features(X_test)
Y_test = preprocess_labels(Y_test)


print("Starting to fit...")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
train_error = np.linalg.norm(Y - clf.predict(X))
print("Train Error: %f" % train_error)

evaluate(clf,X_test,Y_test)