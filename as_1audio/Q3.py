import numpy as np
import matplotlib.pyplot as plt
import scipy
import data_preprocess
import Q2




def phi(C,K):
        t_grid = np.arange(1-0.05*C,0.97,0.05)
        t_grid = np.reshape(t_grid,(t_grid.shape[0],1))
        fea_mat = np.ones((t_grid.shape[0],1))
        for i in range (1,K):
            fea_mat = np.concatenate([fea_mat,t_grid**i],axis=1)
        return fea_mat


def make_vv(C,K):
    square_phi = np.dot(phi(C,K).T,phi(C,K))
    trans_phi = np.matrix.transpose(phi(C,K))
    line  = np.random.randint(0,len(data_preprocess.X_shuf_test))
    phi_t = np.ones((1,K))
    return phi_t.dot(np.dot(np.linalg.inv(square_phi),phi(C,K).T))


'''C = 10 ;K = 3
line = np.random.randint(0,100000)
data = np.array(data_preprocess.X_shuf_train[line]).flatten()[19-C:19]
X = np.reshape(data,(1,data.shape[0]))'''


def vx_(C,K):
    line = np.random.randint(0,100000)
    data = np.array(data_preprocess.X_shuf_train[line]).ravel()
    data2 = np.array(data_preprocess.X_shuf_train[line+1]).ravel()
    X = np.concatenate((data,data2),axis=None)# using 40 random input
    X_c=[]# index of X_c = 41-C from the Cth
    for i in range(0,41-C):
        X_c.append(X[i,i+C])

def Q3c_squareError (C,K):
    line = np.random.randint(0,len(data_preprocess.X_shuf_test))
    data = np.array(data_preprocess.X_shuf_train[line]).flatten()[19-C:19]
    target = data_preprocess.y_shuf_test[line]
    X = np.reshape(data,(1,data.shape[0]))
    fx = np.dot(X,make_vv(C,K).T)
    square_error = (target-fx)**2
    return square_error


def Q3c_showVaries_CK():   #conclusion is best choice for C and K is C= 19 K = 2
    se=[]
    for K in range(2,7):
        cc = np.arange(3,19,1)
        for C in range(3,19):
            strC = str(C)
            strK = str(K)
            print ("C:"+strC+" K:"+strK+"    se: " +str(Q3c_squareError(C,K)))
            print ("\n")
            se.append(Q3c_squareError(C,K))
        plt.plot(cc,np.array(se).flatten())
        se.clear()
    plt.legend(['2','3','4','5','6'])
    plt.show()

def least_square(C):
    K = 3
    square = 0
    data = np.array(data_preprocess.X_shuf_train[0:np.shape(data_preprocess.X_shuf_train)[0]//200-C]).ravel()
    for i in range(C+1,len(data)-C):
        X = data[i:i+C][None,:]
        y = data[i+C:i+C+1][None,:]
        Vt = np.linalg.lstsq(X,y)[0]
        square += (y[0][0]-np.dot(X,Vt)[0])**2
    print (square)

def least_square(C,K):
    square = 0
    data = np.array(data_preprocess.X_shuf_train[0:np.shape(data_preprocess.X_shuf_train)[0]//200-C]).ravel()
    for i in range(C+1,len(data)-C):
        X = data[i:i+C][None,:]
        y = np.dot(X,make_vv(C,K).T)
        square += (y[0][0]-data[i+C:i+C+1][None,:])**2
    print (square)

#6,3   572.4844388
#3,3   2757.8552366
#10,3  338.31
#19,3  225.78
#10,2  208.31
#6,2   267.902
#15,2  185.15
#15,5  743
#19,2  175.114
least_square(19,2)
