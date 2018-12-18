import pandas as pd
from numpy import matrix
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
#loading the data set
df1 = pd.read_csv('C:\Users\Ravi\Desktop\Assignment-1_15110102\iris.csv', sep = ',')


X = df1.ix[:,0:4].values

#Since PCA is an unsupervised method we will scale the data matrix X to have zero mean and unit standard deviation.
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


A = matrix(X_std.transpose())

A1 = A.mean(1)

A2 = A-A1     #subtarcting the mean
r1 =  A2.shape[0] #number of rows of A2
s1 = A2.shape[1] #number of columns of A2

A3 = A2.transpose() #taking transpose of A2
r2 = A3.shape[0] #number of rows of A3
s2 = A3.shape[1] #number of columns of A3

result = [[0 for j in range(r1)] for i in range(s2)]

a2 = A2.tolist()

a3 = A3.tolist()
# Code for computing covariance matrix.


for i in range(r1):
    for j in range(s2):
        for k in range(s1):
           
            
            result[i][j] = result[i][j] + a2[i][k]*a3[k][j]
        result[i][j] = result[i][j]/(s1-1)



Result = matrix(result)


eig_vals, eig_vecs = np.linalg.eig(Result)

new = eig_vecs[:,0:2]


Y =  X_std.dot(new)



Final = Y.tolist()


#plotting the projection matrix.
with plt.style.context('seaborn-whitegrid'):
    for i in range(0,150):
        if i <50:
            
            plt.scatter(Final[i][0], Final[i][1], c = 'blue')
        elif 50 <= i < 100:
            plt.scatter(Final[i][0], Final[i][1], c = 'red')
        elif 100<= i <150:
            plt.scatter(Final[i][0], Final[i][1],  c = 'green')
        if i == 49:
            plt.scatter(Final[i][0], Final[i][1], c = 'blue', label = 'Iris-Setosa')
        if i == 99:
            plt.scatter(Final[i][0], Final[i][1], c = 'red', label = 'Iris-Versicolor' )
        if i == 149:
            plt.scatter(Final[i][0], Final[i][1], c = 'green', label = 'Iris-Virgnica' )
            
            
       
        
            
    
plt.legend(loc='lower center')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA',fontsize = 14, fontweight =  'bold')

plt.tight_layout()

plt.show()






