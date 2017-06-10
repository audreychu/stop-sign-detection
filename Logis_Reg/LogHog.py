import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np


data = pd.read_csv("hog_data.csv")
del data['index']
Y = data['Stop']
X = data.drop('Stop',axis=1)
del X['level_0']


Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = .2)

model = LogisticRegression()
model.fit(Xtrain,Ytrain)
test = model.predict(Xtest)

scores = cross_val_score(model, X, Y, cv=100)
print(np.average(scores))