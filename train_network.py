import numpy as np
from Neural_Network import Layer , Neural_Network
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X , y = dataset.data , dataset.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train,X_test=X_train.T,X_test.T
y_train,y_test=y_train.reshape(1,-1),y_test.reshape(1,-1)

print( X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = Neural_Network(layers = [
    Layer(10,activation='relu'),
    Layer(5,activation='relu'),
    Layer(1,activation='sigmoid')
],lr = 0.1,epochs=5000,data_normalized=False)
clf.fit(X_train,y_train,optimization='momentum_descent')
print(clf.predict (X_test))
print(y_test)
clf.model_eval(X_test,y_test)


# for i in clf.b :
#     print(i.shape)