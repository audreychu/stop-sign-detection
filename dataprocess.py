import pandas as pd
import numpy as np
#import tflearn as tfl
import random
import sklearn
from sklearn.cross_validation import train_test_split
#apparently it's called something else in my version of sklearn
#from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


os.chdir("C://Users/Jeremy/Desktop/School/160")


image_list = []
for path in ['resizedyes/*.jpg', 'resizedno/*.jpg']:
    for filename in glob.glob(path):
        if len(image_list) < 1600:
            im=Image.open(filename)
            c = im.copy()
            image_list.append(c)
            im.close()


#789 are yes
a = np.zeros([1,len(image_list)])
for i in range(789):
    a[0][i] = 1

    
b = a.tolist()[0]
df = pd.DataFrame(image_list, columns=['image'])
df['label'] = b
###END DATA PROCESSING###
#Split train test 
xtrain, xtest, ytrain, ytest = train_test_split(df['image'], df['label'], test_size = 0.2)


convnet = input_data(shape = [None, 640, 436, 3], name = 'input')

convnet = conv_2D(incoming = convnet, nb_filter = 32, filter_size =  2, activation = 'relu', name = 'layer1')
convnet = max_pool_2D(convnet, 2, name = 'maxpool1')

convnet = conv_2D(incoming = convnet, nb_filter = 64, filter_size =  2, activation = 'relu', name = 'layer2')
convnet = max_pool_2D(convnet, 2, name = 'maxpool2')

convnet = conv_2D(incoming = convnet, nb_filter = 128, filter_size =  2, activation = 'relu', name = 'layer3')
convnet = max_pool_2D(convnet, 2, name = 'maxpool3')

convnet = conv_2D(incoming = convnet, nb_filter = 256, filter_size =  2, activation = 'relu', name = 'layer4')
convnet = max_pool_2D(convnet, 2, name = 'maxpool4')

convnet = conv_2D(incoming = convnet, nb_filter = 512, filter_size =  2, activation = 'relu', name = 'layer5')
convnet = max_pool_2D(convnet, 2, name = 'maxpool5')

convnet = fully_connected(convnet, n_units = 1024,activation = 'relu', name = 'fullyconnected1')
convnet = dropout(convnet, keep_prob = 0.8, name = 'dropout')

convnet = fully_connected(convnet, n_units = 2,activation = 'softmax', name = 'fullyconnected2')
convnet = regression(convnet, optimizer = 'adam', loss='categorical_crossentropy', learning_rate=0.001, name = 'regression')


#fits model
model = tflearn.DNN(convnet, tensorboard_dir = '/tmp/tflearn_logs/')
model.fit({'input': X}, {'targets': Y}, n_epoch = 10, snapshot_epoch = True, run_id = 'stopsign', 
          snapshot_steps = '200', validation_set=({'input': test_x}, {'targets': test_y}))

model.save('tflearnsimpleCNN.model')