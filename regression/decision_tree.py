import numpy as np
import tensorflow as tf
from sklearn import tree

# load data
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


X = np.load("proximity_train_features.npy")
Y = np.load("proximity_train_labels.npy")
X = preprocess_features(X)

X_test = np.load("proximity_test_features.npy")
Y_test = np.load("proximity_test_labels.npy")
X_test = preprocess_features(X_test)

print("Starting to fit...")
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,Y)
train_error = np.linalg.norm(Y - clf.predict(X))
print("Train Error: %f" % train_error)

predictions = clf.predict(X_test)
test_error = np.linalg.norm(Y_test - predictions)
print("Test Error: %f" % test_error)

