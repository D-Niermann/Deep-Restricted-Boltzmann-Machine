import tensorflow as tf 
import numpy as np
import os
from math import sqrt
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
def smooth(x,f):
    m=np.ones(f)*1./f
    return np.convolve(x, m,"valid")


def tile(w):
    a=np.zeros([int(sqrt(w.shape[0]*w.shape[1])),int(sqrt(w.shape[0]*w.shape[1]))])
    j,k=0,0
    pic_size=int(sqrt(w.T.shape[1]))
    r=w.T.shape[0]
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
	return 1./(1.+tf.exp(-1./T*x))

def sigmoid_np(x,T):
    return 1./(1.+np.exp(-1./T*x))