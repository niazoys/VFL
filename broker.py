import numpy as np
import random
from guest import Guest
from LR import model
from host import Host
import vertical_splitter as vs
from multiprocessing import Process

#number of participants

#batch size
bs = 100
#number of samples
N = 1000 
#Number of batch 
num_batch=np.ceil(N/bs)
#Communication Round 
comm_round = 10

x1,x2,x3=vs.get_data()
y=vs.get_labels()

guest = Guest(model,data=(x1,y))
host1 = Host(model,data=x2)
host2 = Host(model,data=x3)
seen_sample=[]


def generate_batch_ids(limit=1000,n_samples=200,batch_size=bs):
    ids=[]
    counter=0
    r = random.sample(range(limit), n_samples)
    for e in r:
        if e not in seen_sample:
            seen_sample.append(e)
            ids.append(e)
            counter=counter+1
            if counter==batch_size:
                return ids


for r in range(comm_round):
    for batch in range(num_batch+1):
        ids=generate_batch_ids()


