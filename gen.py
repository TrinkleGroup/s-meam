import numpy as np

uc = np.array([[0,0,0],[.5,.5,.5]])

data = np.empty((1,3))

for i in xrange(6):
    for j in xrange(6):
        for k in xrange(6):
            data = np.concatenate((data, np.array([el+np.array([i,j,k]) for el\
                in uc])))

data *= 4.2
print(data)
with open('data.tmp2','w') as f:
    for i in xrange(1,len(data)+1):
        f.write("%d 1 %f %f %f\n" % (i, data[i-1,0], data[i-1,1], data[i-1,2]))
