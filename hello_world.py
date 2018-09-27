# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:54:27 2018

@author: Shantanu
"""
# IMP scikit-learn uses real-valued features i.e 0,1,2 likewise features
from sklearn import tree
# we use 1 for bumply and 0 for smooth
features = [[140,1], [130,1],[150,0],[170,0]]
# 0 for apples and 1 for oranges
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels )
print(clf.predict([[150,1]]))