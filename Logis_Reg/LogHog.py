import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np


data = pd.read_csv("/Users/audreychu/Documents/4th Year/STA160/hog_data2.csv")
del data['index']
Y = data['Stop']
X = data.drop('Stop',axis=1)
# del X['level_0']


Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = .2, random_state=1)

model = LogisticRegression()
model.fit(Xtrain[Xtrain.columns[:-1]],Ytrain)
test = model.predict(Xtest[Xtest.columns[:-1]])
scores = cross_val_score(model,Xtrain[Xtrain.columns[:-1]],Ytrain,cv=5)


y = list(Ytest)
yhat = list(test)
count = 0
for i in xrange(len(test)):
    if y[i]==yhat[i]:
        count += 1.0
        
print float(count/len(test))


# Evalute score by corss validation
scores = cross_val_score(model,Xtrain[Xtrain.columns[:-1]], Ytrain, cv=100)
print(np.average(scores))
# 0.739331501832


    