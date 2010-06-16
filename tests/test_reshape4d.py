import numpy as np

N=2

a=np.empty((N, N, N, N), dtype=tuple)

for i in xrange(N):
    for j in xrange(N):
        for k in xrange(N):
            for l in xrange(N):
                a[i][j][k][l]=(i, j, k, l)

print a
print 'Reshaped:'
print a.reshape(N*N, N*N)

