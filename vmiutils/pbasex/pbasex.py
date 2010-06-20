import sys
import numpy
import math as m
import _basisfn as bf

def __odd(x):
    return x % 2

def calc_matrix(kmax, lmax, Rbins, Thetabins, sigma=None, oddl=True,
                epsabs=0.0, epsrel=1.0e-7, wkspsize=100000):
    """Calculates an inversion matrix.

    kmax determines the number of radial basis functions (from k=0..kmax).

    lmax determines the maximum value of l for the legendre polynomials
    (l=0..lmax). 

    Rbins specifies the number of radial bins in the image to be inverted.

    Thetabins specifies the number of angular bins in the image to be
    inverted.

    sigma specifes the width of the Gaussian radial basis functions. This is
    defined according to the normal convention for Gaussian functions
    i.e. FWHM=2*sigma*sqrt(2*ln2), and NOT as defined in the Garcia, Lahon,
    Powis paper. If sigma is not specified it is set automatically such that
    the half-maximum of the Gaussian occurs midway between each radial
    function.

    epsabs and epsrel specify the desired integration tolerance when
    calculating the basis functions.

    wkspsize specifies the maximum number of subintervals used for the
    numerical integration of the basis functions.
    """
    kdim = kmax + 1
    ldim = lmax + 1

    # Calculate separation between centres of radial functions
    rwidth = Rbins / kdim
    
    if sigma == None:
        # This sets the FWHM of the radial function equal to the radial
        # separation between radial basis functions
        sigma = rwidth / (2.0 * m.sqrt(2.0 * m.log(2.0)))
        

    # Thetabins is the number of bins used for the range
    # Theta=0..2*Pi. However, the Legendre polynomials have the property
    # that P_l(cos A) = P_l(cos(2Pi-A)), so we can use this symmetry to reduce
    # the computation effort.
    dTheta = 2.0 * numpy.pi / Thetabins

    if __odd(Thetabins):
        midTheta = Thetabins // 2
    else:
        midTheta = Thetabins // 2 - 1

    mtx = numpy.empty((kdim, ldim, Rbins, Thetabins))

    for k in xrange(kdim):
        print k
        rk = rwidth * k;
        for l in xrange(ldim):
            if __odd(l) and oddl == False:
                mtx[k, l, :, :] = 0.0
                continue
            for i in xrange(Rbins):
                R = i # Redundant, but aids readability
                for j in xrange(midTheta):
                    Theta = j * dTheta
                    try:
                        result=bf.basisfn(R, Theta, l, rk, sigma, 
                                          epsabs, epsrel, wkspsize)
                    except RuntimeError:
                        print R, Theta, l, rk, sigma
                        sys.exit(1)
                        
                        mtx[k, l, i, j]=result[0]
                        
                        # Use symmetry to calculate remaining values
                        if __odd(Thetabins):
                            mtx[k, l, i, midTheta + 1:Thetabins] = \
                                mtx[k, l, i, midTheta - 1::-1] 
                        else:
                            mtx[k, l, i, midTheta + 1:Thetabins] = \
                                mtx[k, l, i, midTheta::-1] 
                            
                mtx = mtx.reshape((kdim * ldim, Rbins * Thetabins))
        
    return mtx

