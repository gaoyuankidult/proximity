import numpy as np
import matplotlib.pyplot as plt

def evaluate(clf, X, Y):
    predictions = clf.predict(X)

    error = np.logical_not(predictions == Y)
    false_classification = sum(error)/len(Y)
    false_positive = sum(predictions[error] == 0.0)/len(Y)
    false_negative = sum(predictions[error] == 1.0)/len(Y)

    print("-- Results Test Data --")
    print("False Classification Rate: %f" % false_classification)
    print("False Positive: %f" % false_positive)
    print("False Negative: %f" % false_negative)

    plt.plot(Y)
    plt.plot(predictions)
    plt.show()