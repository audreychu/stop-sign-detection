import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob

import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_2d, conv_2d_transpose, max_pool_2d, avg_pool_2d, upsample_2d, conv_1d, max_pool_1d, avg_pool_1d, residual_block, residual_bottleneck, conv_3d, max_pool_3d, avg_pool_3d, highway_conv_1d, highway_conv_2d, global_avg_pool, global_max_pool
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge

label = str(random.randint(0,100000000))
MODEL_NAME ="INCPT"
PWD = "C:\\Users\\austi\\Dropbox\\STA 160\\stop-sign-detection\\Mini_Inception\\Mini_Inception"
PATH = "C:\\Users\\austi\\Desktop\\"
#os.chdir("C://Users/Jeremy/Desktop/School/160")
os.chdir("D:\\Google Drive\\Other\\misc\\testbatch26k")

yespathlist = []
for p in glob.glob('Yes/*.jpg'):
    yespathlist.append(p)
    
nopathlist = []
for p in glob.glob('No/*.jpg'):
    nopathlist.append(p)
#nopathlist=random.shuffle(nopathlist)
nopathlist = nopathlist[0:len(yespathlist)]

allpath = yespathlist + nopathlist
print(len(yespathlist))
print(len(nopathlist))
print(len(allpath))
labellist = []
image_list = []
for path in allpath:
    for filename in glob.glob(path):
            if len(image_list) < len(yespathlist):
                label = 1
            else:
                label = 0
            im=Image.open(filename)
            c = im.copy()
            image_list.append(np.array(c))
            labellist.append(label)
            im.close()
labellist = np.array(labellist)
labellist = labellist.reshape(-1,1)
enc = sklearn.preprocessing.OneHotEncoder(sparse = False)
enc.fit(labellist)
labellist = enc.transform(labellist)

#789 are yes

###END DATA PROCESSING###
#Split train test validate
xtrain, x2, ytrain, y2 = train_test_split(image_list, labellist, test_size = 0.2,random_state = 1)
xvalid, xtest, yvalid, ytest = train_test_split(x2, y2, test_size = 0.5,random_state=1)


#start of building net
network = input_data(shape=[None, 436, 640, 3])
conv1_7_7 = conv_2d(network, 16, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
pool1_3_3 = batch_normalization(pool1_3_3)
conv2_3_3_reduce = conv_2d(pool1_3_3, 16,1, activation='relu',name = 'conv2_3_3_reduce')
conv2_3_3 = conv_2d(conv2_3_3_reduce, 48,3, activation='relu', name='conv2_3_3')
conv2_3_3 = batch_normalization(conv2_3_3)
pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
inception_3a_1_1 = conv_2d(pool2_3_3, 16, 1, activation='relu', name='inception_3a_1_1')
inception_3a_3_3_reduce = conv_2d(pool2_3_3, 32,1, activation='relu', name='inception_3a_3_3_reduce')
inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 48,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
inception_3a_5_5_reduce = conv_2d(pool2_3_3,4, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 8, filter_size=5, activation='relu', name= 'inception_3a_5_5')
inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 8, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

# merge the inception_3a__
inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)



pool5_7_7 = avg_pool_2d(inception_3a_output, kernel_size=7, strides=1)
pool5_7_7 = dropout(pool5_7_7, 0.4)
loss = fully_connected(pool5_7_7, 2,activation='softmax')
network = regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose = 0,best_checkpoint_path = PWD,tensorboard_dir = "C:\\tmp\\tflearn_logs")


model.fit(xtrain,ytrain, n_epoch = 900, snapshot_epoch = True, validation_set=(xvalid,yvalid),snapshot_step = 500, show_metric = True, run_id = MODEL_NAME,batch_size=50)
model.save(PATH + MODEL_NAME)
    

