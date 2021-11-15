from typing import overload
import numpy as np
from multipledispatch import dispatch
import scipy as sp 
sp.random.seed(12345) 


class model:

    def __init__(self,data,lr=0.01):
        x=data
        self.m,self.n=x.shape
        self.w=sp.random.normal(loc=0.0, scale=1.0, size=(self.n,1))
        self.b=0
        self.lr=lr
        self.dw=None
        self.db=None

    def normalize(self,X):
        m, n = X.shape

        for i in range(n):
            X = (X - X.mean(axis=0))/X.std(axis=0)
        return X

    def sigmoid(self,z):
        return 1.0/(1 + np.exp(-z))
    
    def loss(self,y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss
    
    def gradients(self,X, y, y_hat):
        # Gradient of loss w.r.t weights.
        dw = (1/self.m)*np.dot(X.T, (y_hat - y))
        # Gradient of loss w.r.t bias.
        db = (1/self.m)*np.sum((y_hat - y)) 
        return dw, db
    
    def forward(self,x):
        self.x = self.normalize(x)
        z = (np.dot(self.x, self.w) + self.b)
        return z
    
    def compute_gradient(self,z,y):
        y_hat=self.sigmoid(z)
        self.dw, self.db = self.gradients(self.x, y, y_hat)
        return self.dw,self.db
  
        
    def update_model_(self,dw,db,y):
        self.dw=dw
        self.db=db
        # Updating the parameters.
        self.w -= self.lr*self.dw
        self.b -= self.lr*self.db
        l = self.loss(y, self.sigmoid(np.dot(self.x, self.w) + self.b))
        return l
    def update_model(self,dw,db):
        self.dw=dw
        self.db=db
        # Updating the parameters.
        self.w -= self.lr*self.dw
        self.b -= self.lr*self.db
        
    


    def get_gradients(self):
        if (self.dw and self.db) is not None:
            return self.dw, self.db
        else:
            return None

    def predict(self, X):
        x = self.normalize(X)
        preds = self.sigmoid(np.dot(x, self.w) + self.b)
        pred_class = []
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        return np.array(pred_class)


    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy