import numpy as np
import sys
import math
import sklearn.cluster
from skfeature.utility.construct_W import construct_W
from skfeature.utility.sparse_learning import generate_diagonal_matrix, calculate_l21_norm



#d->n_features n->n_samples c->n_clusters
def fun(v,alpha,lamd,se,X, n_selected_features, **kwargs) -> object:
    """

    @rtype: object
    """
    n_samples, n_features = X.shape

    if 'S' not in kwargs:
        S = construct_W(X)

    else:
        S = kwargs['W']
    X = X.transpose()

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # initialize YP

    if 'F0' not in kwargs:
        if 'n_clusters' not in kwargs:
            print >> sys.stderr, "either F0 or n_clusters should be provided"
        else:
            # initialize F
            n_clusters = kwargs['n_clusters']
            shapet = (n_samples, n_clusters)
            YP= Orthogonal_matrix_initialization(shapet)

    else:
        YP = kwargs['F0']


    # initialize Z
    shapet = (n_features, n_clusters)
    Z = Orthogonal_matrix_initialization(shapet)

    # initialize All 1 matrix
    C = np.identity(n_samples) - (np.ones([n_samples, n_samples])) * (1 / n_samples)


    # initialize D
    D = np.identity(n_features)




    x1 = np.dot(C, X.transpose())
    x2 = np.dot(np.dot(X, C), X.transpose())
    x3 = np.dot(X, C)
    _1N=np.ones((n_samples,1))
    _1N1NT=np.dot(_1N,_1N.transpose())

    #initialize S_
    q= np.array_split(X, v)
    s_ = []
    for i in range(v):
        s_.append(funy(q[i], se))


    max_iter = 1000
    obj = np.zeros(max_iter)
    mrx =S.toarray()
    PHI = 0

    for iter_step in range(max_iter):

        #algorithm 1
        for i in range(n_samples):
            flag = 0
            for j in range(v):
                p = mrx[:, i].reshape(mrx.shape[0], 1)
                p = p - (s_[j])[i]
                if (flag == 0):
                    Ai = p
                    flag = 1
                else:
                    Ai = np.append(Ai, p, 1)

            Ai = np.linalg.inv(np.dot(Ai.transpose(), Ai))


            _1v = np.ones((v, 1))
            if (i == 0):
                W = np.dot(Ai, _1v) / (np.dot(np.dot(_1v.transpose(), Ai), _1v))
            else:
                W = np.append(W, np.dot(Ai, _1v) / (np.dot(np.dot(_1v.transpose(), Ai), _1v)), 1)
        #algorithm 2
        for i in range(n_samples):
            ai=np.zeros((n_samples, 1))
            for j in range(v):
                if(j==0):
                  Bi=(s_[j])[i]
                else :
                  Bi=np.append(Bi, (s_[j])[i], 1)
            for j in range(n_samples):
                  tp = ((YP.transpose()[:, i]).reshape(YP.shape[1],1)) - ((YP.transpose()[:, j]).reshape(YP.shape[1],1))
                  ai[j,0]=np.trace(np.dot(tp.transpose(), tp))
            m=np.dot(Bi,W[:,i].reshape(W.shape[0],1))-(alpha/4)*ai
            sum=0
            for j in range(n_samples):
                tmp=m-np.dot(_1N1NT,m)/n_samples+_1N/n_samples
                mrx[i,j]=max(tmp[j,0]-PHI/2,0)
                sum+=max(PHI-2*tmp[j,0],0)
            PHI=sum/n_samples
        #calculate L
        X_sum = np.array(mrx.sum(axis=1))
        D_ = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            D_[i, i] = X_sum[i]
        L = D_ - mrx


        #algorithm 3,4

        # calculate A
        A = C + alpha * L

        # calculate ~A
        sigma = np.linalg.svd(A,0,0)
        tmp = float(str(max(sigma)))
        A = tmp * np.identity(n_samples) - A


        # calculate B
        B = np.dot(x1, Z)

        # calculate M
        M = np.dot(A, YP) + 2 * B

        # calculate  YP
        U, sigma, VT = np.linalg.svd(M, 0)
        YP= np.dot(U, VT)

        # calculate A2
        A2 = x2 + lamd* D

        # calculate B2
        B2 = np.dot(x3, YP)

        # calculate ~A2
        sigma= np.linalg.svd(A2,0,0)
        tmp = float(str(max(sigma)))
        A2 = tmp * np.identity(n_features) - A2

        # calculate M2
        M2 = np.dot(A2, Z) + 2 * B2

        # calculate  Z
        U, sigma, VT = np.linalg.svd(M2, 0)
        Z = np.dot(U, VT)

        # 更新D
        D = generate_diagonal_matrix(Z)

        obj[iter_step] =calculate_obj(Z,YP,mrx,W,n_samples,X,lamd,alpha,L,s_,v)
        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step + 1, obj[iter_step]))

        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step - 1]) < 1e-3:
            break
    return Z


def Orthogonal_matrix_initialization(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))

    a = np.random.normal(0.0, 1.0, flat_shape)

    u, _, v = np.linalg.svd(a, full_matrices=False)

    q = u if u.shape == flat_shape else v

    return q.reshape(shape)


def kmeans_initialization(X, n_clusters):
    """
    This function uses kmeans to initialize the pseudo label

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_clusters: {int}
        number of clusters

    Output
    ------
    Y: {numpy array}, shape (n_samples, n_clusters)
        pseudo label matrix
    """

    n_samples, n_features = X.shape
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    Y = np.zeros((n_samples, n_clusters))
    for row in range(0, n_samples):
        Y[row, labels[row]] = 1
    T = np.dot(Y.transpose(), Y)
    F = np.dot(Y, np.sqrt(np.linalg.inv(T)))
    F = F + 0.02 * np.ones((n_samples, n_clusters))
    return F


def calculate_obj(Z,YP,mrx,W,n_samples,X,lamd,alpha,L,s_,v):
    C=np.identity(n_samples)-(1/n_samples)*np.ones((n_samples,n_samples))
    t1=np.dot(np.dot(C,X.transpose()),Z)-np.dot(C,YP)
    sum=0
    for i in range(n_samples):
        flag = 0
        for j in range(v):
            p = mrx[:, i].reshape(mrx.shape[0], 1)
            p = p - (s_[j])[i]
            if (flag == 0):
                Ai = p
                flag = 1
            else:
                Ai = np.append(Ai, p, 1)
        tmp=np.dot(Ai,W[:,i].reshape(W.shape[0],1))
        sum+=np.dot(tmp.transpose(),tmp)
    return np.trace(np.dot(t1.transpose(),t1)) + lamd * calculate_l21_norm(Z) + alpha * np.trace(np.dot(np.dot(YP.transpose(), L),YP))+sum

#calculate Sij
def funy(x, alpha):
        n = x.shape[1]
        s = np.zeros((n, n))
        for i in range(n):
            sum = 0.0
            for j in range(n):
                tmp = np.linalg.norm(x[:, i] - x[:, j])
                s[i, j] = math.exp((-tmp * tmp) / alpha)
                sum+=s[i, j]
            for j in range(n):
                s[i,j]=s[i,j]/sum
        return np.split(s, n, 1)