import tensorflow as tf 
import numpy as np
import numpy.random as rnd
import os
from matplotlib.pyplot import savefig
from math import sqrt
def save_fig(path,save_to_file):
    if save_to_file:
        return savefig(path)
    else:
        return None
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

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


def tile(w):
    a        = np.zeros([int(sqrt(w.shape[0]*w.shape[1])),int(sqrt(w.shape[0]*w.shape[1]))])
    j,k      = 0,0
    pic_size = int(sqrt(w.T.shape[1]))
    r        = w.T.shape[0]
    for i in range(r):
        try:
            a[k:k+pic_size,j:j+pic_size]=w.T[i,:].reshape(pic_size,pic_size)
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


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),scale_rows_to_unit_interval=True,output_pixel_vals=True):
    

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def save(name_extension, w=[],bias_v=[],bias_h=[]):
      """ als argument nur eine array in der die ws und bias sind , durch iterieren-> jedes element savetxt()"""
      path="/Users/Niermann/Google Drive/Masterarbeit/Python"
      os.chdir(path)
      if len(w)!=0:
      	np.savetxt("weights-%s.txt"%name_extension, w)
      if len(bias_v)!=0:
      	np.savetxt("bias_v-%s.txt"%name_extension, bias_v)
      if len(bias_h)!=0:
      	np.savetxt("bias_h-%s.txt"%name_extension, bias_h)
      print "saved weights and biases with name_extension=%s"%name_extension

def init_pretrained(name_extension="0.0651765",w=None,bias_v=None,bias_h=None):
	path="/Users/Niermann/Google Drive/Masterarbeit/Python"
	os.chdir(path)
	print "loading from: "+ path
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
	print "loaded %s objects from file"%str(len(m))
	return m

def sigmoid(x,T):
	return 1./tf.add(1.,tf.exp(-tf.multiply(1./T,x)))


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


if __name__ == "__main__":
    w1_new, pos_change = sort_receptive_field(DBM.w1_np)
    w1_unt = untile(w1_new, [784,400])
    w1_test = untile(tile(w1_unt), [784,400]) #works 
    w2_new = sort_by_index(DBM.w2_np, pos_change)
