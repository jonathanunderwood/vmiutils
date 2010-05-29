import numpy
import scipy.integrate

def calc_basis(k, l, i, j):
    """Calculates the value of a basis function, specified by k and l indices,
    corresponding to the point (R, Theta) in the velocity map polar image by
    the Abel integral."""
    pass


def __integrand(r, R, Theta, sigma, rk):
    


def calc_matrix(kmax, lmax, rmax, rbins, thetabins, oddl=True):

    kdim = kmax + 1
    ldim = lmax + 1

    dR = rmax / Rbins
    dTheta = 2.0 * numpy.pi / Thetabins

    sigma = 
    mtx = numpy.empty((kdim, ldim, rbins, thetabins))

    for k in xrange(kdim):
        for l in xrange(ldim):
            for i in xrange(Rbins):
                for j in xrange(Thetabins):
                    R = i * dR
                    Theta = j * dTheta
                    scipy.integrate.quad(__basis_integrand, R, scipy.infty, 
                                         args=(R, Theta, sigma, rk)) 

    mtx = mtx.reshape((kdim * ldim, rbins * thetabins))
        
    return mtx
