import numpy as np
from sklearn import ensemble
import matplotlib.pyplot as plt
from test_data import evaluate

# preprocessing
X_names = [ "handL", "handR", "Pet", "NoPet", "experience", "age", "distance", 
            "angle", "Male","Female"]

def preprocess_features(X):
    continous_features = [5,6,7]
    # normalize
    X[:, continous_features] -= X[:, continous_features].mean(axis=0)
    X[:, continous_features] /= X[:, continous_features].std(axis=0)

    high = np.max(X,axis=0)
    low = np.min(X,axis=0)
    diff = high - low
    diff[diff == 0] = 1.0 # no scaling for singular values
    X = (X - low)/diff
    X = 2*(X - 0.5) # scale to [-1, 1]

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


# Training
print("Starting to fit...")
clf = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=15)
clf = clf.fit(X,Y)
predictions = clf.predict(X)
error = np.logical_not(predictions == Y)
false_classification = sum(error)/len(Y)
print("False Classification Rate: %f" % false_classification)

evaluate(clf,X_test,Y_test)