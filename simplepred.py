
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob
import pickle
import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_2d, conv_2d_transpose, max_pool_2d, avg_pool_2d, upsample_2d, conv_1d, max_pool_1d, avg_pool_1d, residual_block, residual_bottleneck, conv_3d, max_pool_3d, avg_pool_3d, highway_conv_1d, highway_conv_2d, global_avg_pool, global_max_pool
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge


label = str(random.randint(0,100000000))

#os.chdir("C://Users/Jeremy/Desktop/School/160")
MODEL_NAME = 'stopsigntest1.model'
PATH = 'C://Users/Jeremy/Desktop/School/160/heatmap'

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


if os.path.exists('{0}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model Loaded")

else:
    model.fit(xtrain,ytrain, n_epoch = 100, snapshot_epoch = True, run_id = 'stopsigntest1', validation_set=(xvalid,yvalid),snapshot_step = 50, show_metric = True)

        
    model.save('stopsigntest1.model')


Xlist = []
for p in sorted(glob.glob('*.png'), key=os.path.getmtime):
    Xlist.append(p)

X = []
for path in Xlist:
    im=Image.open(path)
    print(path)
    c = im.copy()
    a = np.array(c)
    im.close()
    probs = model.predict([a])
    X.append(probs[0][1])
print(min(X))
with open("heatmaparray1.pickle",'wb') as mat:
    pickle.dump(X,mat)