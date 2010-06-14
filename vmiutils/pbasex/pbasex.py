import numpy
import _basisfn as bf

def calc_matrix(kmax, lmax, Rbins, Thetabins, rmax=None, sigma=None, oddl=True):

    kdim = kmax + 1
    ldim = lmax + 1

    dR = rmax / Rbins
    dTheta = 2.0 * numpy.pi / Thetabins

    rwidth = rmax / kdim

    ##sigma = 
    mtx = numpy.empty((kdim, ldim, rbins, thetabins))

    for k in xrange(kdim):
        rk = rwidth * k;
        for l in xrange(ldim):
            for i in xrange(Rbins):
                for j in xrange(Thetabins):
                    R = i * dR
                    Theta = j * dTheta
                    scipy.integrate.quad(__basis_integrand, R, scipy.infty, 
                                         args=(R, Theta, sigma, rk)) 

    mtx = mtx.reshape((kdim * ldim, rbins * thetabins))
        
    return mtx
