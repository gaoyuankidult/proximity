import numpy as np


data = np.load("sebastian/proximity_data.npy")
data = data[data[:,-2]!=3]
data = data[data[:,-2]!=4]

num_nan = np.sum(np.isnan(data), axis=0)
total = data.shape[0]
print(1- num_nan / float(total))


for x in xrange(5):
    print data[x]

