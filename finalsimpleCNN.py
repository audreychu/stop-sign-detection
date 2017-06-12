import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.cross_validation import train_test_split
#apparently it's called something else in my version of sklearn idk
#from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


os.chdir("C://Users/Jeremy/Desktop/School/160")
#os.chdir("C:\\Users\\Austin Chi\\Google Drive\\Other\\misc")

yespathlist = []
for p in glob.glob('yesphotos/*.jpg'):
    yespathlist.append(p)
    
nopathlist = []
for p in glob.glob('nophotos/*.jpg'):
    nopathlist.append(p)
#nopathlist=random.shuffle(nopathlist)
nopathlist = nopathlist[0:len(yespathlist)]

allpath = yespathlist + nopathlist

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
xtrain, x2, ytrain, y2 = train_test_split(image_list, labellist, test_size = 0.2, random_state=1)
xvalid, xtest, yvalid, ytest = train_test_split(x2, y2, test_size = 0.5, random_state=1)


#start of building net
convnet = input_data(shape = [None, 436, 640, 3], name = 'input')

convnet = conv_2d(incoming = convnet, nb_filter = 8, filter_size =  2, activation = 'relu', name = 'layer1')
convnet = max_pool_2d(convnet, 2, name = 'maxpool1')

convnet = conv_2d(incoming = convnet, nb_filter = 16, filter_size =  2, activation = 'relu', name = 'layer2')
convnet = max_pool_2d(convnet, 2, name = 'maxpool2')

convnet = conv_2d(incoming = convnet, nb_filter = 32, filter_size =  2, activation = 'relu', name = 'layer3')
convnet = max_pool_2d(convnet, 2, name = 'maxpool3')

convnet = conv_2d(incoming = convnet, nb_filter = 64, filter_size =  2, activation = 'relu', name = 'layer4')
convnet = max_pool_2d(convnet, 2, name = 'maxpool4')

convnet = conv_2d(incoming = convnet, nb_filter = 32, filter_size =  2, activation = 'relu', name = 'layer5')
convnet = max_pool_2d(convnet, 2, name = 'maxpool5')

convnet = fully_connected(convnet, n_units = 16,activation = 'relu', name = 'fullyconnected1')
convnet = dropout(convnet, keep_prob = 0.8, name = 'dropout')

convnet = fully_connected(convnet, n_units = 2,activation = 'softmax', name = 'fullyconnected2')
convnet = regression(convnet, optimizer = 'adam', loss='categorical_crossentropy', learning_rate=0.001, name = 'regression')

model = tflearn.DNN(convnet, tensorboard_dir = '/tmp/tflearn_logs/')

#fits model
if os.path.exists('stopsigntest1.model.meta'):
    model.load('stopsigntest1.model')
    print('model loaded')
    model.predict_label(xtest)
else:
    model.fit(xtrain,ytrain, n_epoch = 100, snapshot_epoch = True, run_id = 'stopsigntest11', validation_set=(xvalid,yvalid),snapshot_step = 50, show_metric = True)

        
    model.save('C://Users/Jeremy/Desktop/School/160/stopsigntest1')