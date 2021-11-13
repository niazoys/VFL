import numpy as np
from guest import Guest
from LR import model
from host import Host
import vertical_splitter as vs
from multiprocessing import Process

#batch size
bs = 100
#number of samples
N = 1000 
#Number of batch 
num_batch=np.ceil(N/bs)
#Communication Round 
comm_round = 10

guest = Guest(model)
host1 = Host(model)
host2 = Host(model)

for r in range(comm_round):
    for batch in range(num_batch+1):
        

