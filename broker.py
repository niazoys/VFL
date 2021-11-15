import numpy as np
import random
from guest import Guest
from LR import model
from host import Host
import vertical_splitter as vs
from multiprocessing import Pool, Process
from tqdm import tqdm

#number of participants
num_client=3
#batch size
bs = 100
#number of samples
N = 1000 
#Number of batch 
num_batch=int(N/bs)
#Communication Round 
comm_round = 20

x1,x2,x3=vs.get_data()
y=vs.get_labels()



seen_sample=[]
def generate_batch_ids(limit=1000,n_samples=1000,batch_size=bs):
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
if __name__=="__main__":
    guest = Guest(model,data=(x1,y))
    host1 = Host(model,data=x2)
    host2 = Host(model,data=x3)
    for r in  tqdm(range(comm_round), desc = 'Communication Round'):
        seen_sample=[]
        for batch in range(int(num_batch)):
            ids=generate_batch_ids()

            # Compute the output Z_b for all participants
            #client_fun_pool=[guest.forward,host1.forward,host2.forward]
            #pool = Pool(processes=10)
            '''for c in client_fun_pool:
                pool.map_async(c, ids)'''
            # pool.map_async(host1.forward, ids)
            #p = Process(target=host1.forward,args=(ids,))
            #p.start()
            #p.join()
            # host1.forward(ids)
            #print(host1.send())
            '''guest.receive(host1.send())
            guest.receive(host2.send())
            
            dw,db=guest.compute_gradient() #fix this
            host1.receive((dw,db))
            host2.receive((dw,db))

            #update model
            loss=pool.map_async(guest.update_model)
            pool.map_async(host1.update_model)
            pool.map_async(host2.update_model)'''
            

            guest.forward(ids)
            host1.forward(ids)
            host2.forward(ids)

            guest.receive(host1.send())
            guest.receive(host2.send())
            
            guest.compute_gradient()

            dw,db = guest.send()
            host1.receive((dw,db))
            host2.receive((dw,db))

            guest.update_model()
            host1.update_model()
            host2.update_model()



    x1_test,x2_test,x3_test = vs.get_testdata()
    y_test = vs.get_testlabels()
    #print(x2_test.shape)
    pred_guest= guest.model.predict(x1_test)
    print(guest.model.accuracy(y_test,pred_guest))
    #print(pred_guest)
    pred_host1= host1.model.predict(x2_test)
    print(host1.model.accuracy(y_test,pred_host1))


    '''pred_host2= host2.model.predict(x1_test)
    print(host2.model.accuracy(y_test,pred_host2))'''
    




            
            


