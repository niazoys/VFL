from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)

x1,x2,x3=X[:,:5],X[:,6:13],X[:,14:]
@staticmethod
def get_data():
    return x1,x2,x3
@staticmethod
def get_labels():
    return y
