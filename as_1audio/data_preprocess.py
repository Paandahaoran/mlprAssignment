import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

'''input data with array size of 33713280, y varies in approximate range 0.77 to -0.63'''
mat_contents = sio.loadmat('amp_data.mat')
raw_data = mat_contents.get('amp_data')



'''data matrix split 70% train,15% validation 15% test'''
feature = 21
resize_numbers = np.resize(raw_data,(feature,int(len(raw_data)/feature)))
rows = np.shape(resize_numbers)
resize_rows = np.resize(resize_numbers,(feature,(rows[1]-rows[1]%20)))#20 means divide rows in 5% each
np.random.seed(19961223)
np.random.shuffle(resize_rows)
resize_miniMatrix = np.hsplit(resize_rows,20)


# create traning dataset
X_shuf_train = np.concatenate((resize_miniMatrix[0],resize_miniMatrix[1]),axis=1)
for i in range (2,14):
    X_shuf_train = np.concatenate((X_shuf_train,resize_miniMatrix[i]),axis=1)
X_shuf_train = np.matrix(X_shuf_train).T
y_shuf_train = X_shuf_train[:,20]
X_shuf_train = np.delete(X_shuf_train, -1, axis=1)


# create validation dataset
X_shuf_val = np.concatenate((resize_miniMatrix[14],resize_miniMatrix[15],resize_miniMatrix[16]),axis=1)
X_shuf_val = np.matrix(X_shuf_val).T
y_shuf_val = X_shuf_val[:,20:21]
X_shuf_val = np.delete(X_shuf_val, -1, axis=1)
# create test dataset
X_shuf_test = np.concatenate((resize_miniMatrix[17],resize_miniMatrix[18],resize_miniMatrix[19]),axis=1)
X_shuf_test = np.matrix(X_shuf_test).T
y_shuf_test = X_shuf_test[:,20:21]
X_shuf_test = np.delete(X_shuf_test, -1, axis=1)









#print (np.hsplit(raw_data,20)
#print (np.resize(raw_data,(feature,int(len(raw_data)/feature))))
#histogram in bins 100
#plt.plot(raw_data)
#hist_stuff = plt.hist(raw_data,1000)
#plt.show()
