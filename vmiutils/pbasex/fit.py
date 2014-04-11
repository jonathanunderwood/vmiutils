import logging
import numpy.linalg
import cPickle as pickle

import vmiutils as vmi
import matrix as pbm
from _fit import *

logger = logging.getLogger('vmiutils.pbasex.fit')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

# Implementation note
# --------------------
# The way the matrix is calculated and stored is
# such that the indices are in the order 
# matrix[k, l, Rbin, Thetabin]. 
#
# The matrix is calculated in unitless dimensions i.e. pixel
# number. When fitting an image, the image is necessarily resampled
# onto a (R, Theta) grid of the same dimensions as the matrix. As such
# there is a scaling factor we need to keep track of which relates the
# pixel number in the resampled image, and the fit, to the original
# image dimensions. This is stored as self.rscale. Also, for
# convenience we store the value of the larges radius sampled in the
# original image, self.rmax. When we calculate any property from the
# fitted coefficients, we have to do that calculation in the unitless
# dimensions of the original matrix, so we have to take insto account
# self.rscale.

class PbasexFit():
    def __init__(self):
        self.coef = None
        self.kmax = None
        self.lmax = None
        self.oddl = None
        self.sigma = None
        self.rkstep = None
        self.rfactor = None
        self.rmax = None
        self.rscale = None
        self.__metadata = ['kmax', 'lmax', 'oddl', 'sigma',
                           'rkstep', 'rfactor', 'rmax', 'rscale']

    def fit_data(self, image, matrix, section='whole', lmax=None, oddl=None):
        '''Performs a fit to the data stored in the vmi.PolarImage instance
        image using the PbasexMatrix instance matrix previously calculated.

        lmax specifies the maximum value of to consider when fitting.

        oddl specifies whether or not to consider odd Legendre
        polynomials when fitting.

        section specifies which part of the image to fit. The default
        is "whole" which means the entire image will be
        fit. "negative" specifies that only the image section with
        Theta in the range 0..-Pi will be fit. "positive" specifies
        that only the image section in the range 0..Pi will be fit.

        image must have the same number of Rbins and Thetabins as
        matrix.

        '''

        if not isinstance(image, vmi.PolarImage):
            logger.error("image is not an instance of PolarImage")
            raise TypeError
        elif not isinstance(matrix, pbm.PbasexMatrix):
            logger.error('matrix is not an instance of PbasexMatrix')
            raise TypeError
        elif (image.rbins is not matrix.Rbins) or (image.thetabins is not matrix.Thetabins):
            logger.error("image and matrix do not have compatible dimensions")
            raise TypeError
        
        if oddl is True and matrix.oddl is False:
            logger.error('odd l requested, but matrix not calculated for odd l')
            raise ValueError
        elif oddl is True and matrix.oddl is True:
            mtx = matrix.matrix
        elif oddl is False and matrix.oddl is True:
            mtx = matrix.matrix[:, ::2, :, :]
        elif oddl is None:
            mtx = matrix.matrix
            oddl = matrix.oddl

        if lmax is not None:
            if lmax > matrix.lmax:
                logger.error('requested lmax greater than that of supplied matrix')
                raise ValueError
            else:
                if oddl is True:
                    ldim = lmax + 1
                else:
                    ldim = lmax / 2 + 1
                mtx = mtx[:, 0:ldim, :, :]
        else:
            lmax = matrix.lmax
            if oddl is True:
                ldim = lmax + 1
            else:
                ldim = lmax / 2 + 1

        kdim = matrix.kmax + 1
        Thetabins = matrix.Thetabins
        Rbins = matrix.Rbins

        if section == 'whole':
            # Fit the whole image
            mtx = mtx.reshape((kdim * ldim, Rbins * Thetabins)) 
            img = image.image.reshape(Rbins * Thetabins)
        elif section == 'negative':
            # Fit only the part of the image in the region Theta = -Pi..0
            if _odd(Thetabins):
                endTheta = Thetabins / 2
            else:
                endTheta = (Thetabins / 2) - 1
            halfThetabins = endTheta + 1
            mtx = mtx[:, :, :, 0:endTheta]
            mtx = mtx.reshape((kdim * ldim, Rbins * halfThetabins))
            img = image.image[:, 0:endTheta]
            img = img.reshape(Rbins * halfThetabins)
        elif section == 'positive':
            # Fit only the part of the image in the region Theta = 0..Pi
            startTheta = Thetabins / 2 # Correct for both even and odd Thetabins
            endtheta = Thetabins - 1
            halfThetabins = Thetabins - startTheta 
            mtx = mtx[:, :, :, startTheta:endTheta]
            mtx = mtx.reshape((kdim * ldim, Rbins * halfThetabins))
            img = image.image[:, startTheta:endTheta]
            img = img.reshape(Rbins * halfThetabins)
        else:
            raise NotImplementedError
        
        coef, resid, rank, s = numpy.linalg.lstsq(mtx.transpose(), img)
        # TODO: do something with resid

        coef = coef.reshape((kdim, ldim))

        # If oddl is False we need to expand the coef array to include the odd
        # l values and set them to zero, such that coef is indexed as coef[k,
        # l] rather than coef[k, l/2]. Then all other functions don't have to
        # care about oddl.
        if oddl is False:
            self.coef = numpy.zeros((kdim, lmax + 1))
            for l in xrange(0, lmax + 1, 2):
                self.coef[:, l] = coef[:, l / 2]
        else:
            self.coef = coef

        self.kmax = matrix.kmax
        self.lmax = lmax
        self.oddl = oddl
        self.sigma = matrix.sigma

        # rmax is the largest value of R in the
        # image we have fit - store this for scaling the fit.
        self.rmax = image.r[-1]

        # rkstep holds the spacing between the centres of adjacent Gaussian
        # radial basis functions
        self.rkstep = float(Rbins) / kdim

        # rfactor holds the scaling factor to convert from radial bin number
        # in the fit to actual position in the original image 
        self.rscale = image.r[1] - image.r[0]

    def calc_radial_spectrum(self, rbins=500, rmax=None):
        """Calculate a radial spectrum from the parameters of a fit. Returns a
        tuple (r, intensity) containing the r values and the corresponding
        intensities. 

        rbins determines the number of points in the returned spectrum.

        rmax is the maximum radius to consider, i.e. the spectrum is
        calculated for r=0..rmax. Note: rmax is the desired radial value and
        not the bin number. If rmax is None (default) then the maximum radius
        in the input image is used."""

        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError

        spec = radial_spectrum(rmax/self.rscale, rbins, self.coef, self.kmax,
                               self.rkstep, self.sigma)
        r = numpy.linspace(0.0, rmax, rbins)

        return r, spec

    def cartesian_distribution(self, bins=500, rmax=None):
        """Calculates a cartesian image of the fitted distribution.
        
        bins specifes the number of bins in the x and y dimension to
        calculate.

        rmax specifies the maximum radius to consider in the
        image. This is specified in coordinates of the original image
        that was fitted to.
        """
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError
        
        # Note that the calculation here is done in scaled (pixel)
        # coordinates, not absolute scaled coordinates.
        dist = cartesian_distribution(rmax/self.rscale, bins, self.coef, self.kmax,
                                      self.rkstep, self.sigma, self.lmax)

        x = numpy.linspace(-rmax, rmax, bins)
        y = numpy.linspace(-rmax, rmax, bins)

        return vmi.CartesianImage(x=x, y=y, image=dist)

    def cartesian_distribution_threaded(self, bins=500, rmax=None, 
                                        nthreads=None):
        """Calculates a cartesian image of the fitted distribution using
        multiple threads for speed.
        
        bins specifes the number of bins in the x and y dimension to
        calculate.

        rmax specifies the maximum radius to consider in the
        image. This is specified in coordinates of the original image
        that was fitted to.
        
        nthreads specifies the number of threads to use. If None, then
        we'll use all available cores

        """
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError
        
        xvals = numpy.linspace(-rmax, rmax, bins)
        yvals = numpy.linspace(-rmax, rmax, bins)

        dist = numpy.empty[bins, bins]
        queue = Queue.Queue(0)

        # Here we exploit the mirror symmetry in the y axis
        for x in xvals[bins/2:bins-1]:
            for y in yvals[0:bins-1]:
                queue.put({'x': this_x, 'y': this_y})

        def __worker():
            while not queue.empty():
                job = queue.get()
                x = job['x']
                y = job['y']
                logger.debug('Calculating cartesian distribution at x={0}, y={1}'.format(x, y))
                dist[x, y] = cartesian_distribution_point (
                    x, y, self.coef, self.kmax, self.rkstep, 
                    self.sigma, self.lmax)
                logger.debug('Finished calculating cartesian distribution at x={0}, y={1}'.format(x, y))
                queue.task_done()

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        for i in range(nthreads):
            t = threading.Thread(target=__worker)
            t.daemon = True
            t.start()

        queue.join()

        # Mirror symmetry
        dist[bins/2:0] = dist[bins/2:bins - 1]

        return vmi.CartesianImage(x=x, y=y, image=dist)

    def beta_coefficients(self, rbins=500, rmax=None):
        '''Calculates the beta coefficients for the fit as a function 
        of r up to rmax.
        
        rbins specifies the number of data points calculated

        rmax specifies the maximum radius to consider and is specified
        in fdimensions of the original image that was fitted.
        '''
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError

        if rmax == None:
            rmax = self.rmax

        beta = beta_coeffs(rmax/self.rscale, rbins, self.coef, self.kmax,
                           self.rkstep, self.sigma, self.lmax)

        r = numpy.linspace(0.0, rmax, rbins)
        
        return r, beta

    def dump(self, file):
        fd = open(file, 'w')

        for object in self.__metadata:
            pickle.dump(getattr(self, object), fd, protocol=2)
        numpy.save(fd, self.coef)

        fd.close()

    def load(self, file):
        fd = open(file, 'r')
        
        try:
            for object in self.__metadata:
                setattr(self, object, pickle.load(fd))
            self.coef = numpy.load(fd)
        finally:
            fd.close()

