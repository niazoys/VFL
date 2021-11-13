import numpy as np
from LR import model
from client_interface import ClientInterface

class Guest(ClientInterface):
    def __init__(self,model:model,data):
        self.x,self.y = data
        self.model = model
    
    def create_batch(self,ids):
        return np.array([self.x[id] for id in ids])

    def forward(self,ids):
        x=self.create_batch(ids)
        self.z = self.model.forward(x)

    def receive(self,_z):
        self.z = np.mean(_z,self.z)
    
    def compute_gradient(self):
        self.dw,self.db = self.model.compute_gradient(self.z,self.y)
    
    def send(self):
        return self.dw,self.db
    
    def update_model(self):
        self.loss = self.model.update_model(self.dw,self.db,self.y)

    