import numpy as np

N=3
M = 2
O = 3
P = 2
a=np.empty((N, M, O, P), dtype=tuple)

for i in xrange(N):
    for j in xrange(M):
        for k in xrange(O):
            for l in xrange(P):
                a[i][j][k][l]=(i, j, k, l)

print a
print 'Reshaped:'
b=a.reshape(N*M, O*P)
print b

n =1
m=0
o=1
p=1

print a[n,m,o,p]
print b[n*M+m, o*P+p]
