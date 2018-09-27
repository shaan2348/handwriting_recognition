# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:16:02 2018

@author: Shantanu
"""

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0]) 

test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)  

print(test_target)
print(clf.predict(test_data))