import scipy.io
from skfeature.function import code
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
from skfeature.utility.sparse_learning import feature_ranking
import numpy as np
def main(v,n_cl):
    # load data
    mat = scipy.io.loadmat('../data/lung_maomao.mat')
    X = mat['X'] # data
    X = X.astype(float)
    i=0
   # while(i<X.shape[0]):
    #    X[i] = MaxMinNormalization(X[i])
    #    i=i+1
    y = mat['Y']    # label
    y = y[:, 0]
    labels, counts = np.unique(y, return_counts=True)
    # construct affinity matrix 
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)
    num_fea = n_cl    # specify the number of selected features
    num_cluster =len(labels)   # specify the number of clusters, it is usually set as the number of classes in the ground truth
    # obtain the feature weight matrix
    try:
        Weight = code.fun(v, 10, 0.001, 10, X, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
    except:
        return
    # sort the feature scores in an ascending order according to the feature scores
    idx = feature_ranking(Weight)

    # obtain the dataset on the selected features
    selected_features = X[:, idx[0:num_fea]]
    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = 0
    acc_total = 0
    for i in range(0, 10):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=num_cluster, y=y)
        nmi_total += nmi
        acc_total += acc
    print(str(v)+" "+str(n_cl)+" "+str(float(nmi_total) / 10)+" "+str(float(acc_total) / 10))
    # output the average NMI and average ACC
    #print('NMI:', float(nmi_total) / 10)
    #print('ACC:', float(acc_total) / 10)
def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

if __name__ == '__main__':
    vscle=[2,4,8,16,32,64]
    n_cscle=[20,40,60,80,100,120,140,160,180,200]
    for v in vscle:
        for n_c in n_cscle:
            main(v,n_c)
    print("finish!")


