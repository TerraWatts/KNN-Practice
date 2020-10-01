'''

Practice Exercise: K Nearest Neighbors in Python
Watts Dietrich
Oct 1 2020

The goal of this exercise is to practice use of the k-nearest neighbors algorithm by building a model to evaluate
the condition of cars based on a few attributes.

I use a car evaluation data set obtained from the UCI machine learning repository here:
https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

Six attributes are used:
buying - buying price
maint - price of maintenance
door - number of doors
persons - passenger capacity
lug_boot - size of luggage compartment
safety - safety rating

These are used to predict class, the final rating of the car, which can have 4 values:
unacc (unacceptable)
acc (acceptable)
good
vgood (very good)

I used a 90/10 train/test split.
Using k=9, the model had an accuracy score of 94.8%

The program also prints the predicted and actual rating of the cars in the test set.

'''


import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# read data into pandas dataframe
data = pd.read_csv("car.data")
print(data.head())

# convert non-numeric data to numerical values using sklearn
# create new preprocessing object
le = preprocessing.LabelEncoder()

# get each data column as a list and transform to an array of integers
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

print(buying)

predict = 'class'

# features. zip() combines everything into one list of tuples
x = list(zip(buying, maint, door, persons, lug_boot, safety))
# target / label
y = list(cls)

# split into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train, '\n', y_test)

# new KNN model with k = 9
model = KNeighborsClassifier(n_neighbors=9)

# fit model and print accuracy
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# predicted data
predicted = model.predict(x_test)

# list of car acceptability labels
names = ["unacc", "acc", "good", "vgood"]

# print predicted vs actual results, substitute names (the string labels) for numeric data
# kneighbors can be used to find the k neighbors of a point, show distances
for x in range(len(predicted)):
    #print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    print("Predicted: ", names[predicted[x]], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True) #this function needs a 2d array input
    #print("N: ", n)
