# Functions relating to "Application of Abel inversion in real-time
# calculations for circularly and elliptically symmetric radiation sources",
# Y. T. Cho and S-J Na, Sci. Technol. 16 878-884 (2005).  Note that the i, j
# indices in the Cho and Na paper start from 1 rather than 0. Here we index
# from zero, so the formulas are adjusted accordingly.

import numpy
import math

def area_matrix(dim):
    """
    Calculate and return a numpy array containing the area matrix as defined
    by Cho and Na. The returned square array is of size dim. The array is
    normalized to unit width.
    """
    # Avoid dictionary lookup for functions in loops
    acos = math.acos
    tan = math.tan
    
    # Note that the Cho Na i, j indices in their paper start from 1 rather
    # than 0. Here we index from zero, so the formulas are adjusted
    # accordingly.

    P = numpy.zeros((dim, dim))
    
    # Here we set d = 1 / dim^2 to normalize to a width of 1. This isn't
    # really necessary - we could dispense with this and the factor of 0.5 in
    # the loop
    dd = 1.0 / (dim * dim)
    for i in xrange(dim):
        ii = float(i)
        for j in xrange(i, dim):
            jj = float(j + 1)
            theta = acos(ii / jj)
            P[i, j] = 0.5 * dd * (jj * jj * theta - 
                                  ii * ii * tan(theta))

    S = P.copy()
    S[0:dim - 1, :] -= P[1:dim, :]
    S[:, 1:dim] -= P[:, 0:dim - 1]
    S[0:dim - 1, 1:dim] += P[1:dim, 0:dim - 1]

    return S

if __name__ == "__main__":
    from timeit import Timer

    S = area_matrix(20)
    print S

#    t = Timer("chona.Smatrix(5)", "from __main__ import ChoNa; chona = ChoNa()")

#    print t.timeit()
