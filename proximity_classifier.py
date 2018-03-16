import numpy as np

data = np.load("proximity_data.npy")
data = data[data[:,-2]!=3]
data = data[data[:,-2]!=4]


# read user information
X = np.load('X.npy')
y = np.load('y.npy')
test_X = np.load('test_X.npy')
test_y = np.load('test_y.npy')

y = map(lambda v: v>0.8, y)
test_y = map(lambda v: v>0.8, test_y)


# All classifiers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = [
#    "Nearest Neighbors",
#         "Linear SVM",
#         "RBF SVM",
#         "Gaussian Process",
#         "Decision Tree",
         "Random Forest"
#         "Neural Net",
#         "AdaBoost",
#         "Naive Bayes"

]

classifiers = [
#    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
#    GaussianProcessClassifier(1.0 * RBF(1.0)),
#    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#    MLPClassifier(alpha=1),
#    AdaBoostClassifier(),
#    GaussianNB()
]

from matplotlib import pyplot as plt
for name, clf in zip(names, classifiers):
    clf.fit(X, y)
    score = clf.score(test_X, test_y)
    plt.plot(clf.predict(test_X)[0::])
    plt.plot(test_y[0::])

plt.savefig('test.png')
plt.show()
