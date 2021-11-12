from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)


class model:

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
        m = X.shape[0]
        # Gradient of loss w.r.t weights.
        dw = (1/m)*np.dot(X.T, (y_hat - y))
        # Gradient of loss w.r.t bias.
        db = (1/m)*np.sum((y_hat - y)) 
        return dw, db
    
    def forward(self, x, y):
        m, n = X.shape
        w = np.zeros((n,1))
        b = 0
        y = y.reshape(m,1)
        x = self.normalize(x)
        z = (np.dot(x, w) + b)
        return z
    
    def update_model(self,x,y,z,lr):
        losses = []
        y_hat=self.sigmoid(z)

        self.dw, self.db = self.gradients(x, y, y_hat)

        # Updating the parameters.
        w -= lr*self.dw
        b -= lr*self.db

        l = self.loss(y, self.sigmoid(np.dot(x, w) + b))
        losses.append(l)
        return self.dw,self.db,losses

    def get_gradients(self):
        return self.dw, self.db

    def predict(self, X):

        x = self.normalize(X)

        preds = self.sigmoid(np.dot(x, w) + b)

        pred_class = []
        # if y_hat >= 0.5 --> round up to 1
        # if y_hat < 0.5 --> round up to 1
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        return np.array(pred_class)


    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy