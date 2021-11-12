from LR import model
from client_interface import ClientInterface

class Guest(ClientInterface):
    def __init__(self,model:model,data=None,mode='host'):
        self.mode = mode
        self.data = data
        self.model = model


    def compute_gradient(self):
        return self.model.get_gradient()
    
    def receive(self,data):
        self.z = data
    
    def send(self):
        return self.compute_gradient()
    
    def train(self):
         _,_,self.loss = self.model.train()


    