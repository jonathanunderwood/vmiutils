from __future__ import division
import numpy as np
cimport numpy as np

cdef inline double _max(double a, double b): return a if a >= b else b
cdef inline double _min(double a, double b): return a if a <= b else b


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
def cart2pol_bicubic(np.ndarray[DTYPE_t, ndim=2] image not None,
                     np.ndarray[DTYPE_t, ndim=2] x,
                     np.ndarray[DTYPE_t, ndim=2] y,
                     int rbins, int thetabins, double rmax, 
                     double xc, double yc):

    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    #
    # For the indices, the "int" type is used. This corresponds to a C int,
    # other C types (like "unsigned int") could have been used instead.
    # Purists could use "Py_ssize_t" which is the proper Python type for
    # array indices.

    #TODO: pass centre in as a tuple, and separate into doubles

    if x is None:
        x = np.arange(image.shape[0])

    if y is None:
        y = np.arange(image.shape[1])

    if xc is None:
        xc = 0.5 * (x[0] + x[-1])
    if yc is None:
        yc = 0.5 * (y[0] + y[-1])
    else:
        xc = centre[0]
        yc = centre[1]

    # Calculate minimum distance from centre to edge of image - this
    # determines the maximum radius in the polar image.
    cdef double xsize = _min(abs(x[0] - xc), x[-1] - xc)
    cdef double ysize = _min(abs(y[0] - yc), y[-1] - yc)
    cdef double max_rad = _min(xsize, ysize)

    if rmax is None:
        rmax = max_rad
    elif rmax > max_rad:
        raise ValueError

    # Polar image bin widths
    cdef double rbw = rmax / (radial_bins - 1)
    cdef double thetabw = 2.0 * numpy.pi / (angular_bins - 1)

    # Cartesian image bin widths - assume regularly spaced
    cdef double xbw = x[1] - x[0]
    cdef double ybw = y[1] - y[0]
    
    cdef unsigned int i, j, k, l
    cdef double r, theta, xx, yy
    cdef np.ndarray pimage = np.empty([rbins, thetabins], dtype=DTYPE)
    cdef double pi = np.pi
    
    for i in xrange(rbins):
        r = i * rbw
        for j in xrange(thetabins):
            theta = j * thetabw - pi
            x = r * sin(theta) + xc
            y = r * cos(theta) + yc
            lowx = <int> x
            lowy = <int> y
            dx = x - lowx
            dy = y - lowy
            val = 0.0
            for k in xrange(-1, 3):
                kk = low
                
def double _R(double x):
    cdef double x1 = x + 2
    cdef double x2 = x + 1
    cdef double x3 = x - 1
    cdef double ans = 0.0;

    if x1 > 0:
        ans += x1 ** 3
    if x2 > 0:
        ans =- 4.0 * x2 ** 3
    if x > 0:
        ans += x ** 3
    if x3 > 0:
        ans -= 4.0 * x3 ** 3

    return ans

    cdef DTYPE_t value
    for x in range(xmax):
        for y in range(ymax):
            s_from = int_max(smid - x, -smid)
            s_to = int_min((xmax - x) - smid, smid + 1)
            t_from = int_max(tmid - y, -tmid)
            t_to = int_min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h
