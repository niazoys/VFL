import numpy as np
from LR import model
from client_interface import ClientInterface

class Guest(ClientInterface):
    def __init__(self,model:model,data):
        self.x,self.y = data
        self.model = model(self.x)
        self.z = None
    
    def create_batch(self,ids):
        self.y_=np.array([self.y[id] for id in ids])
        self.y_=self.y_.reshape(len(self.y_),1)
        return np.array([self.x[id] for id in ids])

    def forward(self,ids):
        x=self.create_batch(ids)
        self.z = self.model.forward(x)

    def receive(self,_z):
        self.z = (_z + self.z) / 2
        #self.z = np.mean(_z,self.z)
    
    def compute_gradient(self):
        self.dw,self.db = self.model.compute_gradient(self.z,self.y_)
    
    def send(self):
        return self.dw,self.db
    
    def update_model(self):
        self.loss = self.model.update_model_(self.dw,self.db,self.y_)

    

    