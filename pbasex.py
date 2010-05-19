import numpy
import scipy.integrate

def calc_basis(k, l, ir, itheta):
    """Calculates the value of a basis function specified by k and l indices
    corresponding to the point (r, theta) in the velocity map image by
    the Abel integral."""
    
    scipy.integrate.quad(__basis_integrand, 0, 
    pass


def calc_matrix(kmax, lmax, rmax, rbins, thetabins, oddl=True):
    
    kdim = kmax + 1
    ldim = lmax +1

    mtx = numpy.fromfunction(calc_basis, (kdim, ldim, rbins, thetabins))
    mtx = mtx.reshape((kdim * ldim, rbins * thetabins))
        
    return mtx
