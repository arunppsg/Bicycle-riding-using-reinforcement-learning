#https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('train.csv')
#images = labeled_images.iloc[0:,1:]
images = labeled_images.iloc[0:8000,1:]
#labels = labeled_images.iloc[0:,:1]
labels = labeled_images.iloc[0:8000,:1]


#Above, I take only 8000 for test. Sice if all images where taken, it will consume time.
#For building mode, 8000 images are taken.
#In final model, all images will be taken.

#To get sharper images and making it easy for classification.
i = 1
img = images.iloc[i].values
img = img.reshape((28,28))
plt.imshow(img,cmap = 'gray')
plt.title('Before bit changing '+str(labels.iloc[i,0]))
plt.show()

images[images>0] = 1

i = 1
img = images.iloc[i].values
img = img.reshape((28,28))
plt.imshow(img,cmap = 'gray')
plt.title('After bit changing '+str(labels.iloc[i,0]))
plt.show()


#Making data into training and testing
train_images,test_images,train_labels, test_labels = train_test_split(images,labels,train_size = 0.8, test_size = 0.2, random_state = 0)
