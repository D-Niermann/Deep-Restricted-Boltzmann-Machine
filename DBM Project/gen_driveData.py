import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import exp,sqrt
np.set_printoptions(precision=4)

from Logger import *
log = Logger(True)

def sigmoid(x):
    return 1/(1+np.exp(-x/1))


n_visible = 5
n_hidden = 2
n_data_points = 10000

# define a weight matrix
w = rnd.randn(n_visible, n_hidden)-0.5
# set small elemts to zero
w[np.abs(w)<0.5] = 0

# define the layer vectors
v = np.zeros(n_visible)
h = np.zeros(n_hidden)
train_data = np.zeros([n_data_points, n_visible])
train_label = np.zeros([n_data_points, n_hidden])

log.start("generating Data")
for i in range(n_data_points):
    # define v as random vector
    v = rnd.rand(n_visible)
    # calculate h with the def weights + add some noise
    h = sigmoid(np.dot(v,w) + rnd.random(n_hidden)*0.1)
    # assing the v and h to the data and label
    train_data[i] = v
    train_label[i] = h
log.end()


## show results
print("W")
print(w)
print("\n")

# print distribution of hidden activities
plt.hist(train_label.flatten(),bins=20,lw=0.5,edgecolor="k")
plt.show()
