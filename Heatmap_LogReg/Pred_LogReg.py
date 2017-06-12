import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure


Xlist = []
for p in sorted(glob.glob('/Users/audreychu/Documents/4th Year/STA160/stop-sign-detection/Heatmap_LogReg/New_Images/*png'), key=os.path.getmtime):
    Xlist.append(p)
    

# Calculate HOG
def cal_HOG(path):
    df = pd.DataFrame()
    jpg = cv2.imread(path)
    image = color.rgb2gray(jpg)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    fd = pd.DataFrame(fd)
    df = df.append(fd)
    return df


# Build Model Again
data = pd.read_csv("/Users/audreychu/Documents/4th Year/STA160/hog_data2.csv")
del data['index']
Y = data['Stop']
X = data.drop('Stop',axis=1)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = .2, random_state=1)

model = LogisticRegression()
model.fit(Xtrain[Xtrain.columns[:-1]],Ytrain)
test = model.predict(Xtest[Xtest.columns[:-1]])


X = []
for path in Xlist:
    im=Image.open(path)
    print(path)
    c = im.copy()
#    a = np.array(c)
    im.close()
    hog1 = cal_HOG(path)
    print(hog1.shape)
    probs = model.predict_proba(hog1.T)
    X.append(probs[0][1])
print(min(X))
with open("heatmaparray.pickle",'wb') as mat:
    pickle.dump(X,mat)


