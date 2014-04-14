from __future__ import print_function

import sys
import numpy
import cPickle as pickle
import math as m
import logging
import multiprocessing
import Queue
import threading

import fit as pbfit

# Set up logging and create a null handler in case the application doesn't
# provide a log handler
logger = logging.getLogger('vmiutils.pbasex.matrix_detfn1')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


class PbasexMatrixDetFn1 (PbasexMatrix):
    def calc_matrix_threaded(self, Rbins, Thetabins, kmax, lmax, sigma=None, oddl=True,
                             epsabs=0.0, epsrel=1.0e-7, wkspsize=100000,
                             detectionfn, alpha=0.0, beta=0.0,
                             nthreads=None):
        """Calculates an inversion matrix using multiple threads.

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

        detectionfn specifies a detection function. At present this
        can be None, or an instance of PbasexFit from a previous fit.

        alpha specifies the azimuthal angle between the frame that the
        detection function is specified in and the lab frame. This is
        in radians. If None, we assume 0.0 for this angle.

        beta specifies the polar angle between the frame that the
        detection function is specified in and the lab frame. This is
        in radians. If None, we assume 0.0 for this angle.

        nthreads specifies the number of threads to use. If this has
        the default value of None, the number of threads used will be
        equal to the number of CPU cores.

        """
        if !isinstance(detectionfn, pbfit.PbasexFit):
            raise TypeError('detectionfn is not an instance of PbasexFit')

        # Spacing of radial basis function centres
        rkspacing = Rbins / (kmax + 1.0)

        if sigma is None:
            # If sigma is not specified, we calculate the spacing between the
            # centers of the Gaussian radial functions and set the FWHM of the
            # Gaussians equal to the Gaussian separation
            sigma = rkspacing / (2.0 * m.sqrt(2.0 * m.log(2.0)));

        mtx = numpy.empty([kmax + 1, lmax + 1, Rbins, Thetabins])
        queue = Queue.Queue(0)

        for k in range(kmax + 1):
            for l in range(lmax + 1):
                queue.put({'k': k, 'l': l})

        def __worker():
            while not queue.empty():
                job = queue.get()
                k = job['k']
                l = job['l']

                rk = rkspacing * k

                logger.info('Calculating basis function for k={0}, l={1}'.format(k, l))

                try:
                    bf = basisfn_detfn1 (k, l, Rbins, Thetabins, sigma, rk,
                                         epsabs, epsrel, wkspsize,
                                         detectionfn, _alpha, _beta)
                except IntegrationError as errstring:
                    logger.info(errstring)
                    # Should do something about killing all threads here.
                    raise

                mtx[k, l] = bf
                logger.info('Finished calculating basis function for k={0}, l={1}'.format(k, l))
                queue.task_done()
            
        
        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        for i in range(nthreads):
            t = threading.Thread(target=__worker)
            t.daemon = True
            t.start()

        queue.join()

        self.matrix = mtx
        self.kmax = kmax
        self.sigma = sigma
        self.lmax = lmax
        self.oddl = oddl
        self.Rbins = Rbins
        self.Thetabins = Thetabins
        self.epsabs = epsabs
        self.epsrel = epsrel
