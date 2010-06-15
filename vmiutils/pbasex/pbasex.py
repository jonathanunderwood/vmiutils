import numpy
import math as m
import _basisfn as bf

def basisfn(R, Theta, l, rk, sigma, epsabs=0.0, epsrel=1.0e-6, wkspsize=100000):
    try:
        a=bf.basisfn(R, Theta, l, rk, sigma, epsabs, epsrel, wkspsize)
    except RuntimeError:
        print R, Theta, l, rk, sigma

    return a[0]
    
def calc_matrix(kmax, lmax, Rbins, Thetabins, rmax, sigma=None, oddl=True):

    kdim = kmax + 1
    ldim = lmax + 1

    dR = rmax / Rbins
    dTheta = 2.0 * numpy.pi / Thetabins

    rwidth = rmax / kdim

    if sigma == None:
        # This sets the FWHM of the radial function equal to the radial
        # separation between radial basis functions
        sigma = rwidth / (2.0 * m.sqrt(2.0 * m.log(2.0)))

    mtx = numpy.empty((kdim, ldim, Rbins, Thetabins))

    for k in xrange(kdim):
        print k
        rk = rwidth * k;
        for l in xrange(ldim):
            for i in xrange(Rbins):
                R = i * dR
                for j in xrange(Thetabins):
                    Theta = j * dTheta
                    #print k, l, i, j
                    mtx[k][l][i][j]=basisfn(R, Theta, l, rk, sigma)
                    
    mtx = mtx.reshape((kdim * ldim, Rbins * Thetabins))
        
    return mtx

