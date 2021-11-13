import numpy
from LR import model
from client_interface import ClientInterface

class Guest(ClientInterface):
    def __init__(self,model:model):
        #self.x,self.y = data
        self.model = model

    def forward(self,x):
        self.z = self.model.forward(x)

    def receive(self,_z):
        self.z = numpy.mean(_z,self.z)
    
    def compute_gradient(self):
        self.dw,self.db = self.model.compute_gradient(self.z,self.y)
    
    def send(self):
        return self.dw,self.db
    
    def update_model(self):
        self.loss = self.model.update_model(self.dw,self.db,self.y)

    