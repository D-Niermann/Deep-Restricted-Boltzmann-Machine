import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import exp,sqrt
np.set_printoptions(precision=4)
from Logger import *

log = Logger(True)

n_images = 10000
train_data = np.ones([n_images, 28, 28])

def f(x,y, x_off, y_off, size):
    return exp(-((x-x_off)**2+(y-y_off)**2)/size)

log.start("Generating data")
for i in range(n_images):
    
    x_off = rnd.randint(0,28)
    y_off = rnd.randint(0,28)
    size = rnd.randint(5, 500)
   
    for x in range(28):
        for y in range(28):
            
            train_data[i,x,y] = f(x, y, x_off, y_off, size)
log.end()

print("Generated %i images with values ranging from %f to %f"%
            (
            n_images, 
            train_data.min(),
            train_data.max(),
            )
        )


fig,ax = plt.subplots(1,10, figsize = (16,3))
for i in range(10):
    ax[i].matshow(train_data[i])
plt.tight_layout()
plt.show()