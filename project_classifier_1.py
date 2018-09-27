# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:05:24 2018

@author: Shantanu + Mayank + Anurag
"""

import numpy as np #package used for scientific computing
import matplotlib.pyplot as plt #plotting library for python
#%matplotlib inline
import tensorflow as tf #open source software library for data flow programming across a range of tasks


learn = tf.contrib.learn #contains contributed code that is to be merged in to core Tensor Flow
tf.logging.set_verbosity(tf.logging.ERROR) #sets the threshold for what messages will be logged

mnist = learn.datasets.load_dataset('mnist') #is used to load 'MNIST' datasets
data = mnist.train.images #vairable to load the train images from 'MNIST' data set
labels = np.asarray(mnist.train.labels, dtype=np.int32) #information of train images is copied in labels
test_data = mnist.test.images #variable to load the test images of 'MNIST' datasets
test_labels = np.asarray(mnist.test.labels, dtype=np.int32) #information of test images is copied in test_labels


max_examples = 20000 #Maximum range of index through the 'MNIST' train dataset 
data = data[:max_examples] #data of the defined index is copied to data
labels = labels[:max_examples] #labels matrix is copied to labels variable

#function is defined to display the label and Example in a plot of 28x28 pixel
def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)    
    
# we display some examples for checking correctness of datasets    
display(0)
display(1)
display(8)

# printing length of elements in dataset 
# here length of each element will be 784 as our data is interpreted linearly
print("length of elements in data is:",len(data[0]))

# we are going to train a linear classifier for our data
# here in classifier we are taking 10 features for 10 digits
# feature_columns informs classifier about features we are going to use
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns = feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=1000)

# we will use evaluate function to compare our predicted values and values of labels
# and then use this to calculate our classifier accuracy
# here evaluate and accuracy both are inbuilt in classifier
x = classifier.evaluate(test_data, test_labels)
#print(classifier.evaluate(test_data, test_labels)["accuracy"])
print(x)
print("Accuracy:",x["accuracy"]*100)
