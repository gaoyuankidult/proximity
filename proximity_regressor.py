import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from copy import deepcopy

data = np.load("proximity_data.npy")
data = data[data[:,1]!=3]
data = data[data[:,1]!=4]
data = np.delete(data, -1, 1)  #remove the last column of dataset

# read user information
import csv
user_age = []
user_gender = []
user_pet = []
user_hand = []
user_experience = []

table_UIDs = []
MALE = 1
FEMALE = 0

OWN_PET = 1
NOT_OWN_PET = 0

LEFT_HAND = 1
RIGHT_HAND = 0

with open('users.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        
        table_UIDs.append(row[2])
        
        user_age.append(row[5])

        # user gender
        if 'F' in row[4]:
            user_gender.append(FEMALE)
        elif 'M' in row[4]:
            user_gender.append(MALE)

        # pet ownership
        if 'Y' in row[7]:
            user_pet.append(OWN_PET)
        elif 'N' in row[7]:
            user_pet.append(NOT_OWN_PET)

        # dominant hand
        if 'L' in row[8]:
            user_hand.append(LEFT_HAND)
        elif 'R' in row[8]:
            user_hand.append(RIGHT_HAND)

        # robot experience
        user_experience.append(row[9])



table_UIDs = map(int, table_UIDs[2::])
users_age = np.array(user_age[2::])
users_gender = user_gender
users_pet= user_pet
users_hand= user_hand
users_experience= map(int, user_experience[2::])

print users_experience


# filter data with 0 angle and user ID 1
angles = np.unique(data[:,1])

data[:,1] = data[:,1] * 45.0
angles = np.unique(data[:,1])
UIDs = np.unique(data[:,2])

preprocessed_set = None

for user in UIDs:
    index = table_UIDs.index(user)
    user_age = users_age[index] # user ID starts from 1
    user_experience = users_experience[index]
    user_gender = users_gender[index]
    pet = users_pet[index]
    hand = users_hand[index]

    
    for angle in angles:
        train_set = data[(data[:,1]==angle) & (data [:,2]==user)]


        # at this moment, the titles are organized as
        # distance, angle, gender(male), heigh
        train_set[:,2] = user_gender

        # distance, angle, gender(male),gender(female) height
        user_data = np.insert(train_set, 2, 1-user_gender, axis=1)

        # age, distance, angle, gender(male),gender(female), height
        user_data = np.insert(user_data, 0, user_age, axis=1)

        #experience, age, distance, angle, gender(male),gender(female), height
        user_data = np.insert(user_data, 0, user_experience, axis=1)

        #pet(yes), experience, age, distance, angle, gender(male),gender(female), height
        user_data = np.insert(user_data, 0, pet, axis=1)

        #pet(yes), pet(no), experience, age, distance, angle, gender(male),gender(female), height
        user_data = np.insert(user_data, 1, 1-pet, axis=1)

        #hand(left), pet(yes), pet(no), experience, age, distance, angle, gender(male),gender(female), height
        user_data = np.insert(user_data, 0, hand, axis=1)

        #hand(left), hand(right), pet(yes), pet(no), experience, age, distance, angle, gender(male),gender(female), height
        user_data = np.insert(user_data, 1, 1-hand, axis=1)

        user_data = user_data.reshape(user_data.shape[0],user_data.shape[1],1)


        # generated training examples
        pos_num = user_data.shape[0]

        init_disctance = user_data[0,6] # distance is at 6th position
        end_distance = user_data[-1,6]
        
        neg_num = int(round(pos_num * end_distance /init_disctance))
        neg_data_shape = user_data.shape
        neg_samples = np.zeros((neg_num, neg_data_shape[1],neg_data_shape[2]))



        neg_samples[:,0,:] = hand
        neg_samples[:,1,:] = 1-hand
        neg_samples[:,2,:] = pet
        neg_samples[:,3,:] = 1-pet
        neg_samples[:,4,:] = user_experience
        neg_samples[:,5,:] = user_age
        neg_samples[:,6,:] = np.random.rand(neg_num,1) * end_distance
        neg_samples[:,7,:] = angle
        neg_samples[:,8,:] = user_gender
        neg_samples[:,9,:] = 1-user_gender


        # set hand heightto the last value of hand heigh
        neg_samples[:,-1,:] = user_data[-1,-1,0] 

        # concatenate data together for furthur modelling
        user_data = np.concatenate((user_data, neg_samples), axis=0)

        # generate lable for all training examples
        x_labels = range(len(user_data[:,-1,0]))
        user_data[:,-1,0] = (user_data[:,-1,0] - min(user_data[:,-1,0]))/ (max(user_data[:,-1,0]) - min (user_data[:,-1,0]))
        normalized_heights = deepcopy(user_data[:,-1,0])
        mu1,sigma1 = curve_fit(norm.cdf, x_labels, user_data[:,-1,0], p0=[0,1])[0]
        user_data[:,-1,:] = norm.cdf(x_labels, mu1, sigma1).reshape(user_data[:,-1,:].shape)


        if preprocessed_set == None:
            preprocessed_set= [user_data]
            
        else:
            preprocessed_set.append(user_data)

import random
random.shuffle(preprocessed_set)
dataset = None
for i in xrange(len(preprocessed_set)):
    if i == 0:
        dataset = preprocessed_set[i]
    else:
        dataset = np.concatenate((dataset, preprocessed_set[i]), axis=0)

dataset = dataset.reshape(dataset.shape[0], dataset.shape[1])
original_dataset = deepcopy(dataset)
test_percentage = 0.1
test_num = int(round(dataset.shape[0]*test_percentage))

features = 10
X = dataset[:-test_num,0:features]
y = dataset[:-test_num,-1]


test_X = dataset[-test_num:,0:features]
test_y = dataset[-test_num:,-1]

i_num = 3

#LSTM regressor
import pandas as pd  
from random import random
import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras import optimizers

in_out_neurons = features
out_neurons = 1
hidden_neurons = 25
drate = 0

from cwrnn import ClockworkRNN

model = Sequential()
model.add(GRU(hidden_neurons,input_shape=(in_out_neurons,1)))
model.add(Activation("relu"))

model.add(Dense(out_neurons))
model.add(Activation("linear"))

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
model.compile(loss="mean_squared_error", optimizer='adam')  
fit_X = X.reshape(X.shape[0], X.shape[1],1)
fit_y = y.reshape(y.shape[0],1)
model.fit(fit_X, fit_y, batch_size=32, nb_epoch=i_num, validation_split=0.1, callbacks=[early_stopping])  

fit_test_X = test_X.reshape(test_X.shape[0], test_X.shape[1],1)
fit_test_y = test_y.reshape(test_y.shape[0],1)
predicted = model.predict(fit_test_X)
rmse = np.sqrt(((predicted - fit_test_y) ** 2).mean(axis=0))
plt.plot(test_y)
plt.plot(predicted)
plt.show()
print "LSTM", np.linalg.norm(test_y - predicted.reshape(predicted.shape[0]))

'''

# Linear Regressor

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()

regr.fit(X, y)
results_y= regr.predict(test_X)
print np.linalg.norm(test_y - results_y)


# LASSO regressor

from sklearn import linear_model
regr = linear_model.Lasso(alpha = 0.1)

regr.fit(X,y)
results_y= regr.predict(test_X)
print np.linalg.norm(test_y - results_y)






first_user_index = 797
angle_X = deepcopy(X[:first_user_index,0:features])
angle_X = angle_X.reshape(angle_X.shape[0], angle_X.shape[1], 1)
angle_y = deepcopy(y[:first_user_index])



predicted = model.predict(angle_X)
plt.plot(angle_y)
plt.plot(predicted)
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111)

nth_points = 1
for i in xrange(0,45):
    angle_index = 1
    angle_X[:, angle_index] = i
    angle_X = angle_X.reshape(angle_X.shape[0], angle_X.shape[1], 1)

    
    predicted = model.predict(angle_X)

    cartesian_x = np.cos(angle_X[:, angle_index]/180. * np.pi) * angle_X[:,0]
    cartesian_y = np.sin(angle_X[:, angle_index]/180. * np.pi) * angle_X[:,0]
#    if i == 0:


#        plt.plot(angle_y)
#        plt.plot(predicted)
#        ax.scatter(cartesian_x[::nth_points], cartesian_y[::nth_points], predicted[::nth_points])
#        ax.scatter(cartesian_x.reshape(cartesian_x.shape[0]), cartesian_y.reshape(cartesian_y.shape[0]), predicted.reshape(predicted.shape[0]))
#        ax.scatter(1,1,1)


#plt.show()
'''
