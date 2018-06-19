import tensorflow as tf 
import numpy as np
import numpy.random as rnd
import os
from matplotlib.pyplot import savefig,matshow,colorbar
from math import sqrt




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
        return savefig(path)
    else:
        log.out("Could not save figure!")
        return None


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
    import matplotlib.pyplot as plt
    import time
    N = 4
    R = 2
    IMG_NUM = 921

    a = int(sqrt(DBM.SHAPE[0])) ## image sidelength
    b = int(a/sqrt(N)) ## new img sidelength
    ind = np.zeros([N,(b+R)**2]).astype(np.int) ## which matrix values (x,y) to set to the weight values for each split
    
    for s in range(N):
        m=0
        if s ==0:
            for i in range(0,b+R): #row
                for j in range(i*a,(i*a)+b+R):
                    ind[s][m] = j
                    m+=1
        if s == 1:
            for i in range(b-R,a): # row
                for j in range(i*a,(i*a)+b+R):
                    ind[s][m] = j
                    m+=1
        if s == 2:
            for i in range(0,b+R):
                for j in range((i*a)+b-R,(i*a)+a): #col
                    ind[s][m] = j
                    m+=1
        if s == 3:
            for i in range(b-R,a): #row
                for j in range((i*a)+b-R,(i*a)+a): #col
                    ind[s][m] = j
                    m+=1

    # fig,ax = plt.subplots(1,4)
    # fig_,ax_full = plt.subplots(1,4)
    w      = np.zeros([4,784,100])
    w_full = np.zeros([784,100])

    w[0,ind[0],:25]   = DBM.weights[0]
    w[1,ind[1],25:50] = DBM.weights[0]
    w[2,ind[2],50:75] = DBM.weights[0]
    w[3,ind[3],75:]   = DBM.weights[0]
    for i in range(N):
        w_full+=w[i]

    plt.matshow(w_full.T)
    # h=[[]]*N
    # v=[[]]*N
    # h_full=[[]]*N
    # v_full=[[]]*N
    # for s in range(N):
    #     h[s] = np.dot(test_data[0], w[s])
    #     plt.matshow(h[s].reshape(10,10),vmin=0,vmax=1,cmap="gray")
        
    # v[s] = np.dot(h[s], w[s].T)
    h_full = np.dot(test_data[0], w_full)
    v_full = np.dot(h_full.flatten(), w_full.T)

    # get a matrix that has the right pattern but with oonly 1 and 0
    where = np.where(w_full!=0)
    w_patt = np.copy(w_full)
    w_patt[where] = 1
    plt.matshow(w_patt.T)
    plt.matshow(w_patt[:,0:25].T)

    # ax[s].matshow(v[s].reshape(28,28),cmap="gray")
    v_full.reshape(28,28)[:,b-R:b+R]*=1/2.
    v_full.reshape(28,28)[b-R:b+R,:]*=1/2.
    plt.matshow(v_full.reshape(28,28),cmap="gray")

    plt.matshow(h_full.reshape(10,10),vmin=0,vmax=1,cmap="gray")
    plt.matshow(test_data[0].reshape(28,28),cmap="gray")

    # test_img = test_data[1:1000]#np.round(rnd.random([3,784])*0.55)
    # myplot(test_img[IMG_NUM])
    
    # test_img_new = split_image(test_img,N,R)
    # print test_img_new.shape

    # new_im_shape = int(sqrt(test_img_new.shape[1]))

    # for i in range(N):
    #     plt.matshow(test_img_new[IMG_NUM,:,i].reshape(new_im_shape,new_im_shape))
    #     plt.plot([2,2],[0,28],"r")
    plt.show()