import numpy
import math

class ChoNa:
    def __init__(self):
        pass

    def Smatrix(self, dim):
        # Note that the Cho Na i, j indices in their paper start from 1 rather
        # than 0. Here we index from zero, so the formulas are adjusted
        # accordingly.

        P = numpy.zeros((dim, dim))

        # Avoid dictionary lookup for functions on each iteration
        acos = math.acos
        tan = math.tan
        
        # Here we set d = 1 / dim^2 to normalize to a width of 1. This
        # isn't really necessary - we could dispense with this and the
        # factor of 0.5 in the loop
        dd = 1.0 / (dim * dim)
        for i in xrange(dim):
            ii = float(i)
            for j in xrange(i, dim):
                jj = float(j + 1)
                theta = acos(ii / jj)
                P[i, j] = 0.5 * dd * (jj * jj * theta - 
                                      ii * ii * tan(theta))

        self.S = P.copy()
        self.S[0:dim - 1, :] -= P[1:dim, :]
        self.S[:, 1:dim] -= P[:, 0:dim - 1]
        self.S[0:dim - 1, 1:dim] += P[1:dim, 0:dim - 1]

if __name__ == "__main__":
    from timeit import Timer

    chona = ChoNa()
    chona.Smatrix(20)
    print chona.S

#    t = Timer("chona.Smatrix(5)", "from __main__ import ChoNa; chona = ChoNa()")

#    print t.timeit()
