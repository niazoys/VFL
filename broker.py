from pandas.core import frame
from guest import Guest
from LR import model
from host import Host
import vertical_splitter as vs
from multiprocessing import Process
#batch size
bs = 10 
#number of samples
m = 1000 
guest = Guest(model)
host1 = Host(model)
host2 = Host(model)



