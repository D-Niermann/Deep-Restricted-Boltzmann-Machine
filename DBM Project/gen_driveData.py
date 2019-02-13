import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import exp,sqrt
np.set_printoptions(precision=4)

from Logger import *
log = Logger(True)

def sigmoid(x):
    return 1/(1+np.exp(-x/1))

def generateDriveData(n_v, n_h, N, w ):

    n_visible = n_v
    n_hidden = n_h
    n_data_points = N

    

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
        h = sigmoid(np.dot(v,w) + rnd.random(n_hidden)*0.05)
        # assing the v and h to the data and label
        train_data[i] = v
        train_label[i] = h
    log.end()
    log.info("train_data.shape = ", train_data.shape, "test_data.shape = ", train_label.shape )
    return train_data, train_label

if __name__ == "__main__":
    ## show results
    n_visible = 6
    n_hidden = 2
    # define a weight matrix
    w = rnd.randn(n_visible, n_hidden)*0.5
    # set small elemts to zero
    w[np.abs(w)<0.45] = 0
    plt.matshow(w.T, cmap="gray")
    plt.colorbar()
    plt.figure()

    train_data, train_label = generateDriveData(n_visible, n_hidden, 100, w)

    # print distribution of hidden activities
    plt.hist(train_label.flatten(), bins=20, lw=0.5, edgecolor="k")
    plt.show()
