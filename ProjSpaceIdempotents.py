import numpy as np
n = 1
#one "copy" of R
X = np.array(list(range(1,100))) #np.linspace(-100,100,1000)
X
#An element of Pn
Rx = np.random.choice(X, size=n+1)
v2 = np.array([Rx[0]**2,Rx[0]*Rx[1],Rx[1]**2])
np.sum(v2**2)
mat = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        mat[i,j] = (v2[i]*v2[j])/np.sum(v2**2)
mat@mat 
mat    
v2
Rx