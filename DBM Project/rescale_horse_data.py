import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

os.chdir("/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project/Horse_data")
dir_rescaled = "/Users/Niermann/Google Drive/Masterarbeit/Python/DBM Project/Horse_data_rescaled"
files = os.listdir(os.getcwd())
for f in files:
	print f 
	img = Image.open(f)
	img_arr = np.array(img)

	beg_x,beg_y = np.where(img_arr==255)
	end_x,end_y = np.where(img_arr==255)

	img_arr  = img_arr[beg_x.min():end_x.max(),beg_y.min():end_y.max()]
	img_arr[img_arr>100]  = 255
	img_crop = Image.fromarray(img_arr)
	img_crop = img_crop.resize([64,64])

	img_crop.save(dir_rescaled+"/%s"%f,format="png")
	

plt.show()