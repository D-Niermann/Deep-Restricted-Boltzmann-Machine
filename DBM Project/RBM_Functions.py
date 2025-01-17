import tensorflow as tf 
import numpy as np
import numpy.random as rnd
import os
from matplotlib.pyplot import savefig,matshow,colorbar
from math import sqrt


def follow_neuron_input(layer, neuron_ind, firerates, DBM):
    """
    layer :: layer index where the neuron is
    neuron_ind :: index of neuron to follow the input of
    firerates :: shape [layer, timestep, batchsize, neuron]

    returns :: both inputs over time for each image in batch and each timestep (bottom up first)
    """
    # top down / recurrent
    td_input = np.zeros(shape=(firerates[layer].shape[:-1]))
    # bottom up
    bu_input = np.zeros(shape=(firerates[layer].shape[:-1]))

    for t in range(len(firerates[layer])):
        bu_input[t] = np.dot(firerates[layer-1][t], DBM.w_np[layer-1]) [:,neuron_ind]
        td_input[t] = np.dot(firerates[layer+1][t], DBM.w_np[layer].T) [:,neuron_ind]

    return bu_input, td_input


def plot_noise_examples():
    fig,ax = plt.subplots(1,5)
    m=1
    ax[0].matshow(sample_np(test_data[1]).reshape(28,28))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel(r"0 \%")
    for nf in [0.2,1,2,100]:
        test_data_noise = sample_np(test_data + (rnd.random(test_data.shape)-0.5)*nf)
        ax[m].matshow(test_data_noise[1].reshape(28,28))
        ax[m].set_xticks([])
        ax[m].set_yticks([])
        label = str(np.round(100*np.mean(np.abs(test_data_noise[2]-test_data[2])),0))
        ax[m].set_xlabel(label+r" \%")
        m+=1

def load_logfile(path):
    for files in os.listdir(path):
        if files=="logfile.txt":
            logfile_={}
            with open(path+"logfile.txt","r") as logfile:
                for line in logfile:
                    for i in range(len(line)):
                        if line[i]==",":
                            save=i
                            break
                    value=line[save+1:-1]
                    try:
                        value=float(value)
                    except:
                        try:
                            value_buff = np.fromstring(value[1:-1], sep = ",")
                            if len(value_buff)>0:
                                value = value_buff
                        except:
                            pass
                    logfile_[line[0:save]] = value
    return logfile_

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def vec_len_np(array,axis):
    return np.sqrt(np.sum(np.square(array),axis=axis))

def vec_len(tensor,axis):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor),axis=axis))

def abs_norm(tensor,axis):
    return tf.reduce_sum(tf.abs(tensor),axis=axis)

def abs_norm_np(array,axis):
    return np.sum(np.abs(array),axis=axis)


def save_fig(path,save_to_file):
    if save_to_file:
        return savefig(path, dpi=200)
    else:
        return None

def save_firerates_to_file(firerates,dir_):
    """ 
    saves all np arrays in the firerates list in folder_name.

    firerates :: list of firerates of every layer and every unit (not every timestep!).
    dir :: string of the name the container folder should have with path.
    """
    print("Saving firerates to file!")
    if len(firerates) > 10:
        raise ValueError("Length of Input bigger 10. Wrong array as input?")

    if not os.path.isdir(dir_):
        os.mkdir(dir_)

    for l in range(len(firerates)):
        np.savetxt(dir_+"/Layer%i.txt"%l, firerates[l],fmt='%.3e')


def calc_neuron_hist(neuron_index, activities, neuron_label, fire_thresh, n_classes):
    """ 
    calcs the activation histogram for every neuron in neuron index based on the given activities of that neuron.

    neuron_index    :: array of neurons to compute the hist for
    activities      :: where the activities (averages) for each image are stored,
                        e.g. DBM.layer_save_test, shape [batchsize,neurons]
    neuron_label    :: corresponding labels to activities, e.g. test_label or 
                        subsets of labels for context
    n_classes       :: how many classes are possible, sets the number of hist bins to this number
    """
    hist = [None]*len(neuron_index)
    for n in range(len(neuron_index)):

        # find images that have high fire rates for that neuron 
        w = np.where(activities[:,neuron_index[n]]>fire_thresh)

        # get the corresponding label to the found images
        sublabel = neuron_label[w]
        
        # create a real label vector where not the [[0,1,0,0,0,0,0],[0,1..],...] 
        # is stored but the [2,7,6,9,1,...]
        label = []
        for i in range(len(sublabel)):
            where = np.where(sublabel[i]==1)[0][0]
            label.append(where)
        label = np.array(label).astype(np.float)

        # calc the hist over the label array and add it t the list 
        hist[n] = np.histogram(label, bins = n_classes)[0]

    return hist


def get_layer_label(DBM_type,n_layers,i,short=False):
    """ 
    construct a label string for each layer i. Used for plots.
    """
    label_str = "Layer "+ r"$h^{(%i)}$"%(i)
    if i==0:
        label_str = "Layer "+ r"$v^{(%i)}$"%(1)

    if DBM_type == "DBM":
        if i==n_layers-1:
            label_str = "Layer "+ r"$v^{(%i)}$"%(2)
    elif DBM_type == "DBM_context":
        if i==n_layers-1:
            label_str = "Layer "+ r"$v^{(%i)}$"%(3)
        if i==n_layers-2:
            label_str = "Layer "+ r"$v^{(%i)}$"%(2)
            
    if short:
        return label_str.replace("Layer ","")
    return label_str

def smooth(x,f):
    m=np.ones(f)*1./f
    return np.convolve(x, m,"valid")


def myplot(array,colbar = False):
    s = int(sqrt(len(array)))
    plt.matshow(array.reshape(s,s))
    if colbar:
        plt.colorbar()


def sort_by_index(m,index,axis=0):
    new_m = np.copy(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            new_m[i,j]=m[i,index[i,j]]
    return new_m


def tile_attention(w):
    a        = np.zeros([int(1.2*sqrt(w.shape[0]*w.shape[1])),int(1.1*sqrt(w.shape[0]*w.shape[1]))])
    j,k      = 0,0
    pic_size = 28
    r        = w.shape[1]
    for i in range(r):
        a[k:k+pic_size*2,j:j+pic_size]=w[:,i].reshape(pic_size*2,pic_size)
        k+=pic_size*2+1
        if k+pic_size*2 > a.shape[0]:
            k=0
            j+=pic_size+1
    return a


def tile(w):
    a        = np.zeros([int(sqrt(w.shape[0]*w.shape[1])),int(sqrt(w.shape[0]*w.shape[1]))])
    j,k      = 0,0
    pic_size = int(sqrt(w.shape[0]))
    r        = w.shape[1]
    for i in range(r):
        try:
            a[k:k+pic_size,j:j+pic_size]=w[:,i].reshape(pic_size,pic_size)
            k+=pic_size
            if k+pic_size>a.shape[0]:
                k=0
                j+=pic_size
        except:
            pass
        # j+=28
    return a

def untile(w,shape):
    """ reverse tile() function 
    w     :: matrix to be untiled
    shape :: desired shape 
    """
    a        = np.zeros([shape[0],shape[1]])
    j,k      = 0, 0
    pic_size = int(sqrt(shape[0]))
    r        = shape[1]

    for i in range(r):
            a[:,i] = w[k:k+pic_size,j:j+pic_size].reshape(shape[0])
            k += pic_size
            if k+pic_size>w.shape[0]:
                k = 0
                j += pic_size

    return a


def save(name_extension, w=[],bias_v=[],bias_h=[]):
      """ 
      als argument nur eine array in der die ws und bias sind , durch iterieren-> jedes element savetxt()
      """
      path="/Users/Niermann/Google Drive/Masterarbeit/Python"
      os.chdir(path)
      if len(w)!=0:
        np.savetxt("weights-%s.txt"%name_extension, w)
      if len(bias_v)!=0:
        np.savetxt("bias_v-%s.txt"%name_extension, bias_v)
      if len(bias_h)!=0:
        np.savetxt("bias_h-%s.txt"%name_extension, bias_h)
      print ("saved weights and biases with name_extension=%s"%name_extension)

def init_pretrained(name_extension="0.0651765",w=None,bias_v=None,bias_h=None):
    path="/Users/Niermann/Google Drive/Masterarbeit/Python"
    os.chdir(path)
    print ("loading from: "+ path)
    m=[]
    if (w)!=None:
        w=np.loadtxt("weights-%s.txt"%name_extension)
        m.append(w)
    if (bias_v)!=None:
        bias_v=np.loadtxt("bias_v-%s.txt"%name_extension)
        m.append(bias_v)
    if (bias_h)!=None:
        bias_h=np.loadtxt("bias_h-%s.txt"%name_extension)
        m.append(bias_h)
    print ("loaded %s objects from file"%str(len(m)))
    return m

def sigmoid(x,T):
    return 1.0/tf.add(1.,tf.exp(-tf.multiply(1./T,x)))


def sigmoid_np(x,T):
    return 1./(1.+np.exp(-1./T*x))


def sample_np(x):
    x=np.array(x)
    rng = rnd.random(x.shape)
    x=(x-rng)>=0
    return x


def clamp(x,x_min,x_max):
    if x>x_max:
        x=x_max
    if x<x_min:
        x=x_min
    return x


def shuffle(x,seed):
    """ will shuffle an array using a seed, so the shuffling can be the same for mutlliple 
        arrays. 
    x       :: array to be shuffled (needs to be 1d)
    seed    :: arrays of tuple (shape = Nx2) where each tuple says which elements to swap. 
                These tuple will be applied sequential. Example: seed=rnd.randint(5,size=[6,2])
    """
    y=np.copy(x)
    for i,j in seed:
        _saved_i_=y[i]
        y[i]=y[j]
        y[j]=_saved_i_
    return y

def sort_receptive_field(m):
    """ tries to order every receptive field to its maximum location
    input:
        w :: weights matrix of the first layer 
    returns:
        w_new :: new tiled matrix
        line_change :: index of each line (which is an image here, because the tiled matrix is used)
                         of each image that got moved and where it got moved 
    """
    w       = np.copy(tile(m))
    w_new   = np.zeros(w.shape)
    pos_max = []
    n       = 0
    im_size=int(sqrt(m.shape[0]))
    line_change = []
    m_shape0 = m.shape[0]
    m_shape1 = m.shape[1]
    for i in range(0,w.shape[0],im_size):
        for j in range(0,w.shape[1],im_size):
            true_x = (i + (j * w.shape[0]) ) % m_shape0
            true_y = (j + (i * w.shape[1]) ) % m_shape1
            line = i/28+(j/28 * w.shape[0]/28)
            switch = 0
            maxi   = abs(w[i:i + im_size,j:j + im_size]).max()
            npw    = np.where(abs(w[i:i+im_size,j:j+im_size]) == maxi)
            pos_max.append([i,j,[npw[0][0],npw[1][0]]])

            # calc index where the images needs to be moved
            move_to = [int((pos_max[-1][2][0]*40./im_size)-10)*im_size, 
                        int((pos_max[-1][2][1]*40./im_size)-10)*im_size]
            if maxi>w.max()*0.25:
                move_to[0]=clamp(move_to[0],0,w.shape[0])
                move_to[1]=clamp(move_to[1],0,w.shape[0])
            else:
                # if maimum not big enough just try to move to the mid field
                move_to[0]=w.shape[0]/2
                move_to[1]=w.shape[1]/2

            # iterate many tries
            for k in range(im_size):
                for m in range(im_size):
                    for vor1 in [1,-1]:
                        for vor2 in [-1,1]:

                            if switch==0:
                                        
                                x_to=move_to[0]+vor1*(k*im_size)
                                y_to=move_to[1]+vor2*(m*im_size)
                                #check if place is already used
                                
                                if w_new[x_to:x_to+im_size,y_to:y_to+im_size].mean()==0:
                                    # move image 
                                    w_new[x_to:x_to+im_size,y_to:y_to+im_size]=w[i:i+im_size,j:j+im_size]
                                    # switch for breaking the tries
                                    switch = 1
                                    # save which image got moveed where
                                    line_to = x_to/28+(y_to/28 * w.shape[0]/28)
                                    line_change.append([line,line_to])

            if switch==0:
                n+=1
    return w_new, np.array(line_change)

def sort_by_index(w,line_change):
    """ sort w by index list given from sort_receptive_field() """
    w_new = np.zeros(w.shape)

    for i in line_change:
        move_to=i[1]
        move_from=i[0]
        w_new[move_to] = w[move_from]
    return w_new

def split_image(batch,N,overlap):
    """ splits a image batch in N equal parts (rectangle)
    batch :: batch of images (flattend) with shape [batchsize,n_pixels]
    N :: Number of parts
    overlap:: how many pixels each part overlaps into adjacent parts
    return :: new batch of shape [old_shape,N]
    """ 
    if sqrt(N) != int(sqrt(N)):
        raise ValueError("Number of parts must be quadratic number")



    # image has side length of im_shape
    im_shape = int(np.sqrt(batch.shape[1]))
    if im_shape/sqrt(N)!=int(im_shape/sqrt(N)):
        raise ValueError("Splitting %i pixels by %i gives no integer"%(im_shape,sqrt(N)))
        
    batch = batch.reshape([batch.shape[0],im_shape,im_shape])
    


    # devision point
    dev = int(im_shape / np.sqrt(N))
    # with overlap
    dev_p = dev+overlap
    dev_m = dev-overlap
 
    # new batch that get returned
    # new img size is (dev_p * dev_p)
    new_batch = np.zeros([batch.shape[0],dev_p,dev_p,4])

    ## create sections on which to split the image in both directions (x,y)
    s = [0]
    for i in range(1,int(np.sqrt(N))):
        s.append(int(dev_p*i))
        s.append(int(dev_m*i))
    s.append(im_shape)
    

    # split the images (see how sections are always permuted and only 0,1 - 2,3 - 4,5 sections are used as limits)
    s1 = 0
    s2 = 0
    ss=[]
    for i in range(0,len(s),2):
        for j in range(0,len(s),2):
            ss.append((i,j))
    
    for n in range(N):
        
        s1 = ss[n][0]
        s2 = ss[n][1]
        
        new_batch[:, :, :, n] = batch[:, s[s1]:s[s1+1], s[s2]:s[s2+1]]

    # new_batch[:, :, :, 1] = batch[:, s[2]:s[3], s[2]:s[3]]
    # new_batch[:, :, :, 2] = batch[:, s[0]:s[1], s[2]:s[3]]
    # new_batch[:, :, :, 3] = batch[:, s[2]:s[3], s[0]:s[1]]
    s = new_batch.shape
    new_batch = new_batch.reshape(s[0],s[1]**2,s[-1])
    return new_batch


if __name__ == "__main__":
    pass