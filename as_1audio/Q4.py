import numpy as np
import matplotlib.pyplot as plt
import Q2
import data_preprocess

'''def vx_(C,K):
    line = np.random.randint(0,100000)
    data = np.array(data_preprocess.X_shuf_train[line]).ravel()
    data2 = np.array(data_preprocess.X_shuf_train[line+1]).ravel()
    X = np.concatenate((data,data2),axis=None)# using 40 random input
    X_c=[]# index of X_c = 41-C from the Cth
    for i in range(0,41-C):
        X_c.append(X[i,i+C])'''



def least_square(C):
    K = 3
    square = 0
    data = np.array(data_preprocess.X_shuf_train[0:np.shape(data_preprocess.X_shuf_train)[0]//200-C]).ravel()
    for i in range(C+1,len(data)-C):
        X = data[i:i+C]
        Xin = np.arange(0,C,1)
        y = data[i+C:i+C+1][None,:]
        Xin = np.reshape(Xin,(Xin.shape[0],1))
        A = np.concatenate([np.ones((Xin.shape[0],1)),Xin],axis=1)
        Vt = np.linalg.lstsq(A,X)[0]
        square += (y[0][0]-np.dot(C,Vt)[0])**2
    print (square)



least_square(2)
#for C = 3   1234.3553404829  1/200
#for C = 4   1773.9958483165365  1/200
#for C = 5   2317.766156899743 1/200
#for C = 2   719.9688738575205        1/200
#for C = 10  5089.84821234836  1/200
#for C = 15  7957.140417526505   1/200
#C = 2 is the best length in this module in train data
