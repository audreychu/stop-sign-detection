import pandas as pd
import numpy as np
#import tflearn as tfl
import random
import sklearn
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob

os.chdir("C://Users/Jeremy/Desktop/School/160")


image_list = []
for path in ['resizedyes/*.jpg', 'resizedno/*.jpg']:
    for filename in glob.glob(path): 
        im=Image.open(filename)
        c = im.copy()
        image_list.append(c)
        im.close()

#789 are yes
a = np.zeros([1,len(image_list)])
for i in range(789):
    a[0][i] = 1