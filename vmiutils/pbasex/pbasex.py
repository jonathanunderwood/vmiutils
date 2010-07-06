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
        self.tolerance = None

        # This private attribute is a list containing the variables that should be
        # saved to a file when self.dump is called and read when self.load is called.
        self.__dumpobjs = ['kmax', 'lmax', 'oddl', 'Rbins', 'Thetabins', 'matrix',
                           'epsabs', 'epsrel', 'tolerance']

    def calc_matrix(self, kmax, lmax, Rbins, Thetabins, sigma=None, oddl=True,
                    epsabs=0.0, epsrel=1.0e-7, tolerance=1.0e-7, wkspsize=100000):
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
        calculating the basis functions. The defaults should suffice.

        tolerance specifies the acceptable relative error returned from the
        numerical integration. The default value should suffice.

        wkspsize specifies the maximum number of subintervals used for the
        numerical integration of the basis functions.
        """

        if sigma == None:
            sigma = -1.0

        if oddl:
            oddl = 1
        else:
            oddl = 0


        while True:
            try:
                mtx = matrix(kmax, lmax, Rbins, Thetabins, sigma, oddl, 
                             epsabs, epsrel, tolerance, wkspsize)
                break
            except MaxIterError:
                logger.info("Maximum integration iterations exceeded")
                raise
            except RoundError:
                logger.error("Round-off error during integration")
                raise
            except SingularError:
                logger.error("Singularity in integration")
                raise
            except DivergeError:
                logger.error("Divergence in integration")
                raise
            except RuntimeError:
                logger.error("Runtime error during matrix calculation")
                raise

        self.matrix = mtx
        self.kmax = kmax
        self.lmax = lmax
        self.oddl = oddl
        self.Rbins = Rbins
        self.Thetabins = Thetabins
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.tolerance = tolerance

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

    
## How to reshape, reminder
# kdim = kmax + 1
# ldim = lmax + 1
# self.matrix = mtx.reshape((kdim * ldim, Rbins * Thetabins)) 
