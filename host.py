import numpy
from LR import model
from client_interface import ClientInterface

class Host(ClientInterface):
    def __init__(self,model:model):
        #self.x = data
        self.model = model

    def forward(self,x):
        self.z = self.model.forward(x)

    def receive(self,grad):
        self.dw,self.db=grad
    
    def send(self):
        return self.z
    
    def update_model(self):
        self.loss = self.model.update_model(self.dw,self.db)

    