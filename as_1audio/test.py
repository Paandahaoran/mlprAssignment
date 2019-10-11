import numpy as np
import matplotlib.pyplot as plt
import scipy
import Q3

data = np.array([1,5,5,435,345,7,7,76,5,3,5,35,3])


i=2
C = 5
X = data[i:i+C][None,:]
y = np.dot(X,Q3.make_vv(C,3).T)
Vt = np.linalg.lstsq(X,y,rcond=-1)[0]
square =(y[0][0]-np.dot(X,Vt)[0])**2



print (np.dot(X,Vt)[0][0])
print(y)
print(y[0][0])
print(square)
