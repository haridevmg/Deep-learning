import numpy as np
import matplotlib.pyplot as plt

# Shallow neural network with 4 nodes in hidden layer and 1 in the output layer

# Activation functions for neural network
def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def l_relu(beta,z):
    return np.maximum(beta*z,z)

def linear(z):

    return z

def softmax(z):
    assert(z.shape[1] == 1)
    t = np.exp(z)
    a = t / np.sum(t)

    return a
class Shallow_NN:

    def __init__(self,lr = 0.01,num_iter = 3000):
        self.lr = lr
        self.num_iter = num_iter

    def fit(self,X_train,y_train):

        n,m = X_train.shape
        self.W1 = np.random.randn(4,n) * 0.01
        self.W2 = np.random.randn(1,4) * 0.01
        self.b1 = np.zeros((4,1))
        self.b2 = np.zeros((1,1))

        for i in range(self.num_iter) :

            Z1 = np.dot(self.W1,X_train)+self.b1
            A1 = relu(Z1)
            Z2 = np.dot(self.W2,A1)+self.b2
            A2 = sigmoid(Z2)

            assert(A2.shape == y_train.shape)

            dA2 = np.multiply(A2,1 - A2)
            dZ2 = A2 - y_train
            dW2 = (1/m) * np.dot(dZ2,A1.T)
            db2 = (1/m) * np.sum(dZ2,axis = 1,keepdims=True)

            dA1 = np.reshape([0 if a<0 else 1 for i,a in np.ndenumerate(A1)],A1.shape)
            dZ1 = np.multiply(np.dot(self.W2.T,dZ2),dA1)
            dW1 = (1/m) * np.dot(dZ1,X_train.T)
            db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)

            assert(A1.shape == dA1.shape and A2.shape == dA2.shape)
            assert(Z1.shape == dZ1.shape and Z2.shape == dZ2.shape)
            assert(self.W1.shape == dW1.shape and self.W2.shape == dW2.shape)
            assert(self.b1.shape == db1.shape and self.b2.shape == db2.shape)

            self.W1 -= self.lr * dW1
            self.W2 -= self.lr * dW2
            self.b1 -= self.lr * db1
            self.b2 -= self.lr * db2

    def predict(self,X_pred,class_names=[]):

        Z1 = np.dot(self.W1,X_pred)+self.b1
        A1 = relu(Z1)
        Z2 = np.dot(self.W2,A1)+self.b2
        y_pred = sigmoid(Z2)

        for y in range(len(y_pred[0])):
            if y_pred[0,y]>=0.5 : y_pred[0,y] = 1
            else : y_pred[0,y] = 0

        return y_pred
    
    def model_eval(self,X_test,y_test):

        y_pred = self.predict(X_test)
        comb = y_pred == y_test

        tp,tn,fp,fn=0,0,0,0
        for i in range(len(comb[0])):

            if comb[0,i] :
                if y_pred[0,i]==1 : tp += 1
                else : tn += 1
            else :
                if y_pred[0,i]==1 : fp +=1
                else : fn += 1

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * precision * recall / (precision+recall)

        print(f"Accuracy of model : {accuracy*100} %")
        print(f"Precision of model : {precision}")
        print(f"Recall of model : {recall}")
        print(f"F1 score of model : {f1}")

class Layer : # creating subclass Layer for each layer in the neural network

    def __init__ (self,no_nodes = 1,activation = "sigmoid"):

        self.no_nodes = no_nodes
        self.activation = activation

    def activate(self,Z_l):
        
        if self.activation == 'sigmoid':
            return sigmoid(Z_l)

        elif self.activation == 'tanh' :
            return tanh(Z_l)

        elif self.activation == 'relu':
            return relu(Z_l)

        elif self.activation == 'leaky relu':
            return l_relu(0.01,Z_l)

    def forward_function(self,A_prev,W,b,keep_rate,dropout = False ):
        
        Z_l = np.dot(W,A_prev)+b
        A_l = self.activate(Z_l)
        if dropout:
            D_l = np.random.rand(A_l.shape[0],A_l.shape[1]) <= keep_rate
            A_l = np.multiply(A_l,D_l)
            A_l /= keep_rate

        return Z_l , A_l
    
    def d_activate(self,Z_l,act):
        
        if act == 'sigmoid':
            A_l = sigmoid(Z_l)
            return np.multiply(A_l,(1 - A_l))
        
        elif act == 'tanh' :
            A_l = tanh(Z_l)
            return 1 - A_l**2
        
        elif act == 'relu':
            A_l = relu(Z_l)
            return np.reshape([0 if a1<0 else 1 for i1,a1 in np.ndenumerate(A_l)],A_l.shape)

        elif act == 'leaky relu':
            A_l= l_relu(0.01,Z_l)
            return np.reshape([0.01 if a2<0 else 1 for i2,a2 in np.ndenumerate(A_l)],A_l.shape)
        

    def backward_function(self,act,dZ_l,Z_prev,A_prev,W_l,m):

        dZ = np.multiply(np.dot(W_l.T,dZ_l),self.d_activate(Z_prev,act))
        dW = (1/m) * np.dot(dZ,A_prev.T) + self.lambd/m * W_l
        db = (1/m) * np.sum(dZ,axis=1,keepdims = True)
        

        return dZ,dW,db
    
class Neural_Network :

    def __init__(self,layers = [ Layer(1,'sigmoid') ] , lr=0.01 , epochs=1000 ,lambd = 0.1 , keep_rate = 1,data_normalized = False, b_1 = 0.9,b_2 = 0.99,eps = 10**-8):

        assert(keep_rate >= 0 and keep_rate <= 1) # Make it mandatory that keep probability is always b/w 0 & 1

        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.lambd = lambd
        self.keep_rate = keep_rate
        self.mean = None
        self.variance = None
        self.norm = data_normalized
        self.eps = eps
        self.b_1 = b_1
        self.b_2 = b_2

# Optimization algorithms :

    def gradient_descent (self,dW,db,W,b):

        W -= self.lr * dW
        b -= self.lr * db

        return W , b
    def momentum(self,dW,db,W,b):
        self.VdW = self.b_1 * self.VdW + (1 - self.b_1)*dW
        self.Vdb = self.b_1 * self.Vdb + (1 - self.b_1)*db 

        W -= self.lr * self.VdW
        b -= self.lr * self.Vdb

        return W,b 
    def rmsprop(self,dW,db,W,b):
        self.SdW = self.b_2 * self.SdW + (1 - self.b_2)*dW
        self.Sdb = self.b_2 * self.Sdb + (1 - self.b_2)*db

        W -= self.lr * dW/(np.sqrt(self.SdW)+self.eps)
        b -= self.lr * db/(np.sqrt(self.Sdb)+self.eps)

        return W,b
    def adam (self,dW,db,W,b,t):

        self.VdW = self.b_1 * self.VdW + (1 - self.b_1)*dW
        self.Vdb = self.b_1 * self.Vdb + (1 - self.b_1)*db
        self.SdW = self.b_2 * self.SdW + (1 - self.b_2)*dW
        self.Sdb = self.b_2 * self.Sdb + (1 - self.b_2)*db
        self.VdW /= 1 - self.b_1 ** t
        self.Vdb /= 1 - self.b_1 ** t
        self.SdW /= 1 - self.b_2 ** t 
        self.Sdb /= 1 - self.b_2 ** t

        W -= self.lr * self.VdW/(np.sqrt(self.SdW))
        b -= self.lr * self.Vdb/(np.sqrt(self.Sdb))

        return W,b

    def normalise_data(self , X_train):
        # Data normalization

        m = X_train.shape[1]
        self.mean = np.sum(X_train , axis = 1 , keepdims = True)/m
        self.variance = np.sum(X_train ** 2 , axis = 1 , keepdims = True)/m
        X_train -= self.mean
        X_train /= np.sqrt(self.variance)

        return X_train

    def fit(self,X_train,y_train,batch_size = None,optimization = 'gradient_descent'):
        
        if self.norm is False : X_train = self.normalise_data(X_train)

        n0,m_b = X_train.shape
        if batch_size is None : batch_size = m_b
        assert(m_b % batch_size == 0)
        self.W , self.b = [],[]
        
        self.layers =  [Layer(n0,activation="None")] + self.layers
        self.no_layers =len(self.layers)

        for l in range(1,self.no_layers):
            self.W.append(np.random.randn(self.layers[l].no_nodes,self.layers[l-1].no_nodes) * np.sqrt(2/self.layers[l-1].no_nodes))
            self.b.append(np.zeros((self.layers[l].no_nodes,1)))

        train_batch = np.reshape(X_train , (m_b//batch_size , n0 , batch_size))
        print(train_batch.shape)
        m = batch_size
        for i in range(self.epochs):

            for b in train_batch :

                Z ,A = [None]*(self.no_layers - 1),[b]+[None]*(self.no_layers - 1)

                for l in range(self.no_layers-1):

                    Z_l , A_l = self.layers[l+1].forward_function(A[l],self.W[l],self.b[l],self.keep_rate,dropout = True)
                    Z[l] ,A[l+1] = Z_l , A_l

                dZ = [A[-1] - y_train]
                dW_L = (1/m) * np.dot(dZ[0],A[-2].T) + self.lambd/m * self.W[-1]
                db_L = (1/m) * np.sum(dZ[0],axis=1,keepdims=True)
                self.VdW = np.zeros(self.W[-1].shape)
                self.Vdb = np.zeros(self.b[-1].shape)
                self.SdW = np.zeros(self.W[-1].shape)
                self.Sdb = np.zeros(self.b[-1].shape)
                if optimization == "gradient_descent":
                    self.W[-1] , self.b[-1] = self.gradient_descent(dW_L,db_L,self.W[-1],self.b[-1])
                elif optimization == 'adam ':
                    self.W[-1] , self.b[-1] = self.adam(dW_L,db_L,self.W[-1],self.b[-1],i)
                elif optimization == 'momentum_descent':
                    self.W[-1] , self.b[-1] = self.momentum(dW_L,db_L,self.W[-1],self.b[-1])
                elif optimization == 'rmsprop':
                    self.W[-1] , self.b[-1] = self.rmsprop(dW_L,db_L,self.W[-1],self.b[-1])

                for l in range(self.no_layers-2,0):
                    
                    dZ_prev , dW_l , db_l = self.layers[l].backward_function(self.activation[l-1],dZ[0],Z[l-1],A[l-1],self.W[l],m)
                    dZ.insert(0,dZ_prev)
                    self.VdW = np.zeros(self.W[l-1].shape)
                    self.Vdb = np.zeros(self.b[l-1].shape)
                    self.SdW = np.zeros(self.W[l-1].shape)
                    self.Sdb = np.zeros(self.b[l-1].shape)
                    if optimization == "gradient_descent":
                        self.W[l-1] , self.b[l-1] = self.gradient_descent(dW_l,db_l,self.W[l-1],self.b[l-1])
                    elif optimization == 'adam ':
                        self.W[l-1] , self.b[l-1] = self.adam(dW_l,db_l,self.W[l-1],self.b[l-1],i)
                    elif optimization == 'momentum_descent':
                        self.W[l-1] , self.b[l-1] = self.momentum(dW_l,db_l,self.W[l-1],self.b[l-1])
                    elif optimization == 'rmsprop':
                        self.W[l-1] , self.b[l-1] = self.rmsprop(dW_l,db_l,self.W[l-1],self.b[l-1])

                # def backward_function(self,dZ_l,Z_prev,A_prev,W_l,m):

                #     dZ = np.multiply(np.dot(W_l.T,dZ_l),self.d_activate(Z_prev))
                #     dW = (1/m) * np.dot(dZ,A_prev.T)
                #     db = (1/m) * np.sum(dZ,axis=1,keepdims = True)
                #     return dZ,dW,db

    def predict(self,X_pred):
        if self.norm is False :
            X_pred -= self.mean
            X_pred /= np.sqrt(self.variance)
        Z ,A = [None]*(self.no_layers - 1),[X_pred]+[None]*(self.no_layers - 1)

        for l in range(self.no_layers-1):

            Z_l , A_l = self.layers[l+1].forward_function(A[l],self.W[l],self.b[l],self.keep_rate)
            Z[l] ,A[l+1] = Z_l , A_l
        y_pred = A[-1]
        if self.layers[-1].activation == 'sigmoid':
            for y in range(len(y_pred[0])):
                if y_pred[0,y]>=0.5 : y_pred[0,y] = 1
                else : y_pred[0,y] = 0

        return y_pred
    
    def user_predict(self,X,class_names=[]):
        y_pred = self.predict(X)
        class_pred = np.zeros(y_pred.shape)
        if class_names != []:
            for i in range(self.no_layers-1):
                class_pred[0,i] = class_names[int(y_pred[0,i])]
        return y_pred
    
    def model_eval(self,X_test,y_test):

        y_pred = self.predict(X_test)
        comb = y_pred == y_test

        tp,tn,fp,fn=0,0,0,0
        for i in range(len(comb[0])):

            if comb[0,i] :
                if y_pred[0,i]==1 : tp += 1
                else : tn += 1
            else :
                if y_pred[0,i]==1 : fp +=1
                else : fn += 1

        accuracy= (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2 * precision * recall / (precision+recall)

        print(f"Accuracy of model : {accuracy*100} %")
        print(f"Precision of model : {precision}")
        print(f"Recall of model : {recall}")
        print(f"F1 score of model : {f1}")

            
            

            