import sys
import numpy
import pickle
import math as m
import logging
from _basisfn import *

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

logger = logging.getLogger('vmiutils.pbasex.pbasex')
__null_handler = __NullHandler()
logger.addHandler(__null_handler)

def _odd(x):
    return x & 1

class PbasexMatrix():
    def __init__(self):
        self.matrix = None
        self.kmax = None
        self.lmax = None
        self.oddl = None
        self.Rbins = None
        self.Thetabins = None
        self.epsabs = None
        self.epsrel = None
        # This private attribute is a list containing the variables that should be
        # saved to a file when self.dump is called and read when self.load is called.
        self.__dumpobjs = ['kmax', 'lmax', 'oddl', 'Rbins', 'Thetabins',
                           'epsabs', 'epsrel', 'matrix']

    def calc_matrix(self, kmax, lmax, Rbins, Thetabins, sigma=None, oddl=True,
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
        rwidth = float(Rbins) / kdim

        if sigma == None:
            # This sets the FWHM of the radial function equal to the radial
            # separation between radial basis functions
            sigma = rwidth / (2.0 * m.sqrt(2.0 * m.log(2.0)))

        # Thetabins is the number of bins used for the range
        # Theta=0..2*Pi. However, the Legendre polynomials have the property
        # that P_l(cos A) = P_l(cos(2Pi-A)), so we can use this symmetry to reduce
        # the computation effort.
        dTheta = 2.0 * numpy.pi / Thetabins

        if _odd(Thetabins):
            midTheta = Thetabins // 2
        else:
            midTheta = (Thetabins // 2) - 1

        mtx = numpy.empty((kdim, ldim, Rbins, Thetabins))

    # Thoughts on removing the loop overhead here - use meshgrid and vectorize?

        for k in xrange(kdim):
            rk = rwidth * k;
            for l in xrange(ldim):
                if _odd(l) and oddl == False:
                    mtx[k, l, :, :] = 0.0
                    continue
                for i in xrange(Rbins):
                    R = i # Redundant, but aids readability
                    for j in xrange(midTheta):
                        Theta = j * dTheta
                        while True:
                            try:
                                result = basisfn(R, Theta, l, rk, sigma, 
                                                 epsabs, epsrel, wkspsize)
                                mtx[k, l, i, j] = result[0]
                                break
                            except MaxIterError:
                                logger.info("Maximum integration iterations exceeded")
                                logger.info("Increasing max iterations by factor of 10")
                                wkspize = 10 * wkspize
                            except RoundError:
                                logger.error("Round-off error during integration")
                                raise
                            except bf.SingularError:
                                logger.error("Singularity in integration")
                                raise
                            except bf.DivergeError:
                                logger.error("Divergence in integration")
                                raise

                        # Use symmetry to calculate remaining values
                        if _odd(Thetabins):
                            mtx[k, l, i, midTheta + 1:Thetabins] = \
                                mtx[k, l, i, midTheta - 1::-1] 
                        else:
                            mtx[k, l, i, midTheta + 1:Thetabins] = \
                                mtx[k, l, i, midTheta::-1] 

        self.matrix = mtx.reshape((kdim * ldim, Rbins * Thetabins))
        self.kmax = kmax
        self.lmax = lmax
        self.oddl = oddl
        self.Rbins = Rbins
        self.Thetabins = Thetabins
        self.epsabs = epsabs
        self.epsrel = epsrel

    def dump(self, file):
        fd = open(file, 'w')

        for object in self.__dumpobjs:
            pickle.dump(getattr(self, object), fd)

        fd.close()

    def load(self, file):
        fd = open(file, 'r')

        for object in self.__dumpobjs:
            setattr(self, object, pickle.load(fd))

        fd.close()
        print 'ding', self.kmax

    def invert_image(self, image):
        pass

class PBasexFit():
    def __init__(self):
        coef = None

    

