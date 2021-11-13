import numpy
from LR import model
from client_interface import ClientInterface

class Host(ClientInterface):
    def __init__(self,model:model,data):
        self.x = data
        self.model = model

    def create_batch(self,ids):
        return np.array([self.x[id] for id in ids])

    def forward(self,ids):
        x=self.create_batch(ids)
        self.z = self.model.forward(x)

    def receive(self,grad):
        self.dw,self.db=grad
    
    def send(self):
        return self.z
    
    def update_model(self):
        self.loss = self.model.update_model(self.dw,self.db)

    