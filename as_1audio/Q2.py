import numpy as np
import  matplotlib.pyplot as plt
import data_preprocess
t = np.arange(0,1,0.05)
y = np.array(data_preprocess.X_shuf_train[int(np.random()*100)].flatten())
target = np.array(data_preprocess.y_shuf_train[77].flatten())




def phi_linear(Xin):
    Xin = np.reshape(Xin,(Xin.shape[0],1))# row to column
    return np.concatenate([np.ones((Xin.shape[0],1)),Xin],axis=1)

def phi_quadratic(Xorig):
    Xorig = np.reshape(Xorig,(Xorig.shape[0],1))#row to column
    return np.concatenate([np.ones((Xorig.shape[0],1)),Xorig,Xorig**2,Xorig**3,Xorig**4],axis=1)


def fit_and_plot(phi_fn,X,yy):
    w_fit = np.linalg.lstsq(phi_fn(X),yy)[0]
    #w_fit=np.dot(np.linalg.pinv(phi_fn(X)),yy)
    x_grid = np.arange(0,1,0.01)
    f_grid = np.dot(phi_fn(x_grid),w_fit)
    plt.plot(x_grid,f_grid,lw=2)



plt.plot(t,y[0],'bD')
fit_and_plot(phi_linear,t,y[0])
fit_and_plot(phi_quadratic,t,y[0])
plt.plot(1,target,'rD')
plt.legend(['traning points','line fit','quadratic fit','test point'],loc='lower left')
plt.show()
