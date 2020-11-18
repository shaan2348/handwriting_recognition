from sklearn.datasets.base import get_data_home

get_data_home()

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

import pandas as pd
pixels = pd.DataFrame(mnist.data)
labels = pd.DataFrame(mnist.target)

pixels.loc[0].values

labels.loc[0].values

import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
label = labels.loc[0]
pixel = pixels.loc[0]
pixel = np.array(pixel, dtype='uint8')
pixel = pixel.reshape((28,28))
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixel, cmap='gray')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7.0)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_pred, y_test)*100)
print()
print(X_test[1124])
print()
print(y_test[1124])


y_predicted = model.predict(X_test[1124].reshape(1,-1))

label = y_predicted
pixel = X_test[1124]
pixel = np.array(pixel, dtype='uint8')
pixel = pixel.reshape((28,28))
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixel, cmap='gray')
plt.show()
