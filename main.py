#https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images,test_images,train_labels, test_labels = train_test_split(images,labels,train_size = 0.8, test_size = 0.2, random_state = 0)

i = 1
img = train_images.iloc[i].values
img = img.reshape((28,28))
plt.imshow(img,cmap = 'gray')
plt.title(train_labels.iloc[i,0])
plt.show()
