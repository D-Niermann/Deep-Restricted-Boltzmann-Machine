import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from math import exp,sqrt
np.set_printoptions(precision=4)
from Logger import *

log = Logger(True)


def generateGradients(n_images):
    train_data = np.ones([n_images, 784])

    def f(x,y, x_off, y_off, size):
        return exp(-((x-x_off)**2+(y-y_off)**2)/size)

    log.start("Generating data")
    for i in range(n_images):
        
        x_off = rnd.randint(0,28)
        y_off = rnd.randint(0,28)
        size = rnd.randint(5, 500)

        m = 0
        for x in range(28):
            for y in range(28):   
                train_data[i,m] = f(x, y, x_off, y_off, size)
                m += 1

    log.end()

    print("Generated %i images with values ranging from %f to %f"%
                (
                n_images, 
                train_data.min(),
                train_data.max(),
                )
            )

    return train_data

if __name__ =="__main__":
    train_data = generateGradients(10)
    fig,ax = plt.subplots(1,10, figsize = (16,3))
    for i in range(10):
        ax[i].matshow(train_data[i])
    plt.tight_layout()
    import seaborn
    plt.matshow(train_data[0])
    
    plt.show()