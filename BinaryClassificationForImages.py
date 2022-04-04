# Import the libraries we'll use below.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns  # for nicer plots
sns.set(style='darkgrid')  # default style
import statistics
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from keras import metrics
from keras.datasets import fashion_mnist

# Load the Fashion MNIST dataset.
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# Flatten Y_train and Y_test, so they become vectors of label values.
# The label for X_train[0] is in Y_train[0].
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

np.random.seed(0) # For reproducibility purposes

# Shuffle the order of the training examples.
indices = np.arange(X_train.shape[0])
shuffled_indices = np.random.permutation(indices)

X_train = X_train[shuffled_indices]
Y_train = Y_train[shuffled_indices]

# Show the data shapes.
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

# Pixel values range from 0 to 255. To normalize the data, we just need to 
# divide all values by 255.
X_train = X_train / 255
X_test = X_test / 255

label_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# Show the first 5 training labels.
print('First 5 label values:', Y_train[0:5])
print('Mapped to their names:', [label_names[i] for i in Y_train[0:5]])

# Create a figure with subplots. This returns a list of object handles in axs
# which we can use populate the plots.
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10,5))
for i in range(5):
  image = X_train[i]
  label = Y_train[i]
  label_name = label_names[label]
  axs[i].imshow(image, cmap='gray')  # imshow renders a 2D grid
  axs[i].set_title(label_name)
  axs[i].axis('off')
plt.show()

fig, axs = plt.subplots(nrows=10, ncols=5, figsize=(10,5))

tshirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, ankle_boots , all= ([] for i in range(11))
total = 0

for i in range(len(Y_train)):

  image = X_train[i]
  label = Y_train[i]
  label_name = label_names[label]

  if (total == 50):
    break
  
  if (len(tshirts) < 5 and label_name == 't-shirt'):
    tshirts.append([image, label_name])
    total = total + 1
    
  if (len(trousers) < 5 and label_name == 'trouser'):
    trousers.append([image, label_name])
    total = total + 1
    
  if (len(pullovers) < 5 and label_name == 'pullover'):
    pullovers.append([image, label_name])
    total = total + 1
    
  if (len(dresses) < 5 and label_name == 'dress'):
    dresses.append([image, label_name])
    total = total + 1
    
  if (len(coats) < 5 and label_name == 'coat'):
    coats.append([image, label_name])
    total = total + 1
   
  if (len(sandals) < 5 and label_name == 'sandal'):
    sandals.append([image, label_name])
    total = total + 1
    
  if (len(shirts) < 5 and label_name == 'shirt'):
    shirts.append([image, label_name])
    total = total + 1
    
  if (len(sneakers) < 5 and label_name == 'sneaker'):
    sneakers.append([image, label_name])
    total = total + 1
    
  if (len(bags) < 5 and label_name == 'bag'):
    bags.append([image, label_name])
    total = total + 1
    
  if (len(ankle_boots) < 5 and label_name == 'ankle boot'):
    ankle_boots.append([image, label_name])
    total = total + 1

all.append(tshirts)
all.append(trousers)
all.append(pullovers)
all.append(dresses)
all.append(coats)
all.append(sandals)
all.append(shirts)
all.append(sneakers)
all.append(bags)
all.append(ankle_boots)

    
for i in range (5): 
  for j in range (10): 
    axs[j][i].imshow(all[j][i][0], cmap='gray')  # imshow renders a 2D grid
    axs[j][i].set_title(all[j][i][1])
    axs[j][i].axis('off')


plt.show()

# Make copies of the original labels.
Y_train_binary = np.copy(Y_train)
Y_test_binary = np.copy(Y_test)

# Update labels: 1 for sneaker images; 0 for the rest.
# Note that a boolean array is created when Y_train_binary != 7 is evaluated.
Y_train_binary[Y_train_binary != 7] = 0.0 
Y_train_binary[Y_train_binary == 7] = 1.0
Y_test_binary[Y_test_binary != 7] = 0.0
Y_test_binary[Y_test_binary == 7] = 1.0

sneakers = X_train[Y_train_binary == 1]
non_sneakers = X_train[Y_train_binary != 1]

array_sneakers_center_value = []
array_non_sneakers_center_value = []
array_sneakers_value_314 = []
array_non_sneakers_value_314 = []

#mean: sum of the terms / number of terms
for i in range (len(sneakers)):
  #gets individual image
  sneaker_image = sneakers[i]
  non_sneaker_image = non_sneakers[i]

  #add value of pixel [14][14] to an array
  array_sneakers_center_value.append(sneaker_image[14][14])
  array_non_sneakers_center_value.append(non_sneaker_image[14][14])
  #add value of pixel [3][14] to an array
  array_sneakers_value_314.append(sneaker_image[3][14])
  array_non_sneakers_value_314.append(non_sneaker_image[3][14])

#calculatates mean
print("[14, 14] Sneaker mean: ", statistics.mean(array_sneakers_center_value))
print("[14, 14] Non-sneaker mean: ", statistics.mean(array_non_sneakers_center_value))
print("[3, 14] Sneaker mean: ", statistics.mean(array_sneakers_value_314))
print("[3, 14] Non-sneaker mean: ", statistics.mean(array_non_sneakers_value_314))

print("[14, 14] Sneaker standard deviation: ", statistics.stdev(array_sneakers_center_value))
print("[14, 14] Non-sneaker standard deviation: ", statistics.stdev(array_non_sneakers_center_value))
print("[3, 14] Sneaker standard deviation: ", statistics.stdev(array_sneakers_value_314))
print("[3, 14] Non-sneaker standard deviation: ", statistics.stdev(array_non_sneakers_value_314))

print('Number of sneaker images in training set:', (Y_train_binary == 1).sum())
print('Number of non-sneaker images in training set:', (Y_train_binary == 0).sum())

X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)
print(X_train.shape)
print(X_test.shape)

X_train = X_train[0:3000]
Y_train_binary = Y_train_binary[0:3000]
Y_test = Y_test[0:3000]
X_test = X_test[0:3000]
Y_test_binary = Y_test_binary[0:3000]

def knn(X_train, Y_train_binary, X_test, k):
  pred = np.zeros(X_test.shape[0])

  for i in range(X_test.shape[0]):
    x_test_example = X_test[i]

    distances = np.sqrt(np.sum((x_test_example - X_train)**2, axis=1))
    idx_nei = np.argsort(distances)[:k]
    distances = distances[idx_nei]
    y_nei = Y_train_binary[idx_nei]

    w = 1 / distances
    w = w/np.sum(w)

    pred[i] = np.dot(w, y_nei)
  return pred

def accuracy(Y_test_binary, pred):
      return np.sum(Y_test_binary == pred) / len(Y_test_binary)

def precision(Y_test_binary, pred):
  return np.sum(Y_test_binary * pred) / np.sum(pred)

def recall(Y_test_binary, pred):
  return np.sum(Y_test_binary * pred) / np.sum(Y_test_binary)

pred = knn(X_train, Y_train_binary, X_test, 5)


print("Accuracy: ", accuracy(Y_test_binary, pred))
print("precision: ", precision(Y_test_binary, pred))
print("recall", recall(Y_test_binary, pred))

for th in [0.1, 0.25, 0.5, 0.75]:
  y_hat = (pred >= th)* 1.0
  print("th: ", th, "y_hat: ", y_hat)

Y_train_binary[Y_train_binary != 7] = 1.0 
Y_train_binary[Y_train_binary == 7] = 0.0 
pred_ns = knn(X_train, Y_train_binary, X_test, 5)
X_test = X_test.reshape(X_test.shape[0],28,28)

erroneous_sneakers = []
erroneous_non_sneakers = []

#sneaker images
y_sneaker_th = (pred >= 0.5)* 1.0 

# gets images of incorrect classified images
for i in range(len(y_sneaker_th)):
  if (len(erroneous_sneakers) < 5):
    if (y_sneaker_th[i] == 1 and Y_test_binary[i] == 0):
      erroneous_sneakers.append(X_test[i])

#gets predicted for non_sneakers
y_non_sneaker_th = (pred_ns >= 0.5)* 1.0 

# gets images of incorrect classified images
for i in range(len(y_non_sneaker_th)):
  if (len(erroneous_non_sneakers) < 5):
    if (y_non_sneaker_th[i] == 1 and Y_test_binary[i] == 0):
      erroneous_non_sneakers.append(X_test[i])
print(len(erroneous_non_sneakers))

#displays images for sneakers
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10,5))
for i in range(5):
  image = erroneous_sneakers[i]
  axs[i].imshow(image, cmap='gray')  # imshow renders a 2D grid
  axs[i].axis('off')
plt.show()

#displays images for non_sneakers
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10,5))
for i in range(5):
  image = erroneous_non_sneakers[i]
  axs[i].imshow(image, cmap='gray')  # imshow renders a 2D grid
  axs[i].axis('off')
plt.show()

X_test = X_test.reshape(X_test.shape[0],-1)
X_train = X_train.reshape(X_train.shape[0],-1)
Y_test_binary[Y_test_binary != 7] = 0.0
Y_test_binary[Y_test_binary == 7] = 1.0

def most_common(labels):
    counter = np.zeros(np.amax(labels) +1 )
    for c in labels:
        counter[c] = counter[c] + 1
    return np.argmax(counter)

def dist(a0, a1):
  return np.sqrt(np.sum((a0-a1) ** 2))

def knn_multiclass(X_train, Y_train, X_test, k):
  pred = np.zeros(X_test.shape[0])
  d = np.zeros(X_train.shape[0])

  for i in range(X_test.shape[0]): #traverses through test examples
    test_example = X_test[i] 

    for j in range(X_train.shape[0]):#traverses through training examples
      train_example = X_train[j] 
      d[j] = dist(test_example, train_example)# gets distances

    #gets the k distances
    nei = np.argsort(d)[:k]
    pred[i] = most_common(Y_train[nei]) # gets the most repeating labels

  return pred

knn_multiclass(X_train, Y_train, X_test, 5)

k3_pred = knn_multiclass(X_train, Y_train, X_test, 3)
k6_pred = knn_multiclass(X_train, Y_train, X_test, 6)
k7_pred = knn_multiclass(X_train, Y_train, X_test, 7)
k4_pred = knn_multiclass(X_train, Y_train, X_test, 4)
k10_pred = knn_multiclass(X_train, Y_train, X_test, 10)

cm1 = confusion_matrix(Y_test, k3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp.plot()
plt.show()

cm2 = confusion_matrix(Y_test, k6_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp.plot()
plt.show()

cm3 = confusion_matrix(Y_test, k7_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp.plot()
plt.show()

cm4 = confusion_matrix(Y_test, k4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm4)
disp.plot()
plt.show()

cm5 = confusion_matrix(Y_test, k10_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm5)
disp.plot()
plt.show()

print("k3 Accuracy: ", accuracy(Y_test, k3_pred))
print("k6 Accuracy: ", accuracy(Y_test, k6_pred))
print("k7 Accuracy: ", accuracy(Y_test, k7_pred))
print("k4 Accuracy: ", accuracy(Y_test, k4_pred))
print("k10 Accuracy: ", accuracy(Y_test, k10_pred))
