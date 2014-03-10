from __future__ import print_function

import sys
import numpy
import cPickle as pickle
import math as m
import logging
from _matrix import *

# Set up logging and create a null handler in case the application doesn't
# provide a log handler
logger = logging.getLogger('vmiutils.pbasex.matrix')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

def _odd(x):
    return x & 1

class PbasexMatrix():
    def __init__(self):
        self.matrix = None
        self.kmax = None
        self.sigma = None
        self.lmax = None
        self.oddl = None
        self.Rbins = None
        self.Thetabins = None
        self.epsabs = None
        self.epsrel = None

        # This private attribute is a list containing the variables that should be
        # saved to a file when self.dump is called and read when self.load is called.
        self.__metadata = ['Rbins', 'Thetabins', 'kmax', 'sigma', 'lmax', 'oddl',
                           'epsabs', 'epsrel']

    def calc_matrix(self, Rbins, Thetabins, kmax, lmax, sigma=None, oddl=True,
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
        calculating the basis functions. The defaults should suffice.

        tolerance specifies the acceptable relative error returned from the
        numerical integration. The default value should suffice.

        wkspsize specifies the maximum number of subintervals used for the
        numerical integration of the basis functions.
        """

        if sigma is None:
            # If sigma is not specified, we calculate the spacing between the
            # centers of the Gaussian radial functions and set the FWHM of the
            # Gaussians equal to the Gaussian separation
            spacing = float(Rbins) / (kmax + 1) 
            sigma = spacing / (2.0 * m.sqrt(2.0 * m.log(2.0)));

        while True:
            try:
                mtx = matrix(kmax, lmax, Rbins, Thetabins, sigma, oddl, 
                             epsabs, epsrel, wkspsize, None)
                break
            except IntegrationError as errstring:
                logger.info(errstring)
                raise

        self.matrix = mtx
        self.kmax = kmax
        self.sigma = sigma
        self.lmax = lmax
        self.oddl = oddl
        self.Rbins = Rbins
        self.Thetabins = Thetabins
        self.epsabs = epsabs
        self.epsrel = epsrel

    def dump(self, file):
        fd = open(file, 'w')

        for object in self.__metadata:
            pickle.dump(getattr(self, object), fd, protocol=2)
        numpy.save(fd, self.matrix)

        fd.close()

    def load(self, file):
        fd = open(file, 'r')
        
        try:
            for object in self.__metadata:
                setattr(self, object, pickle.load(fd))
            self.matrix = numpy.load(fd)
        finally:
            fd.close()

    def print_params(self):
        for object in self.__metadata:
            print('{0}: {1}'.format(object, getattr(self, object)))
                
