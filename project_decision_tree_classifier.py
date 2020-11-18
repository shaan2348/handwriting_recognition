#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# loading the train.csv file as matrix
# we are going to use the same dataset for both testing and training purpose
data = pd.read_csv("datasets/train.csv").as_matrix()
# initialising the classifier
clf = DecisionTreeClassifier()

#training dataset
xtrain = data[0:21000, 1:]
train_label = data[0:21000,0]

clf.fit(xtrain, train_label)

#testing data
xtest = data[21000:,1:]
actual_label = data[21000:,0]

# d = xtest[8]
# d.shape = [28,28]
# pt.imshow(255-d, cmap = 'gray')
# print(clf.predict([xtest[0]))
# pt.show()

# storing the predicted value of test set
p = clf.predict(xtest)

count = 0
for i in range(0,21000):
    count+=1 if p[i] == actual_label[i] else 0
print("Accuracy =",(count/21000)*100)
