from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1600, n_features=60, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)

x1,x2,x3=X[:1000,:20],X[:1000,20:40],X[:1000,40:]
x1_test,x2_test,x3_test= X[1000:,:20],X[:1000,20:40],X[1000:,40:]

y_train = y[:1000]
y_test = y[1000:]


def get_data():
    return x1,x2,x3
    
def get_labels():
    return y_train

def get_testdata():
    return x1_test,x2_test,x3_test

def get_testlabels():
    return y_test



