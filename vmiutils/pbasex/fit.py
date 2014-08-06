import logging
import numpy.linalg
import cPickle as pickle
import Queue
import threading
import multiprocessing
import math

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
#
# matrix[k, l, Rbin, Thetabin].
#
# The matrix is calculated in unitless dimensions i.e. pixel
# number. When fitting an image, the image is necessarily resampled
# onto a (R, Theta) grid of the same dimensions as the matrix. As such
# there is a scaling factor we need to keep track of which relates the
# pixel number in the resampled image, and the fit, to the original
# image dimensions. For convenience we store the value of the largest
# radius sampled in the original image, self.rmax. When we calculate
# any property from the fitted coefficients it is necessary to ensure
# we scale to the dimensions in the original image. So, once fittd, we
# store sigma, rkstep etc rescaled to the original image.


class PbasexFit(object):

    def __init__(self):
        self.coef = None
        self.kmax = None
        self.lmax = None
        self.oddl = None
        self.sigma = None
        self.rkstep = None
        self.rmax = None
        self.__metadata = ['kmax',
                           'lmax',
                           'oddl',
                           'sigma',
                           'rkstep',
                           'rmax']

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
            logger.error(
                'odd l requested, but matrix not calculated for odd l')
            raise ValueError
        elif oddl == matrix.oddl:
            mtx = matrix.matrix
        elif oddl is False and matrix.oddl is True:
            mtx = matrix.matrix[:, ::2, :, :]
        elif oddl is None:
            mtx = matrix.matrix
            oddl = matrix.oddl

        if lmax is not None:
            if lmax > matrix.lmax:
                logger.error(
                    'requested lmax greater than that of supplied matrix')
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
            # Correct for both even and odd Thetabins
            startTheta = Thetabins / 2
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

        # rbinw is the scaling factor to convert from radial bin
        # number in the fit to actual position in the original image -
        # used here to scale rkstep and sigma to image coordinates
        rbinw = float(image.r[1] - image.r[0])

        # We need to store sigma in image dimensions.
        self.sigma = matrix.sigma * rbinw

        # rmax is the largest value of R in the image we have fit -
        # store this for scaling the fit. Note that we need to add a
        # single bin width to image.r[-1], since r[-1] is the lowest r
        # value in the final bin.
        self.rmax = image.r[-1] + rbinw

        # rkstep holds the spacing between the centres of adjacent
        # Gaussian radial basis functions.
        self.rkstep = rbinw * Rbins / kdim

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

        spec = radial_spectrum(rmax, rbins, self.coef, self.kmax,
                               self.rkstep, self.sigma)

        # Calculate r values. Set enpoint=False here, since the r
        # values are the lowest value of r in each bin.
        r = numpy.linspace(0.0, rmax, rbins, endpoint=False)

        return r, spec

    def cartesian_distribution_threaded(self, bins=250, rmax=None,
                                        truncate=5.0, nthreads=None,
                                        weighting='normal'):
        """Calculates a cartesian image of the fitted distribution using
        multiple threads for speed.

        bins specifes the number of bins in the x and y dimension to
        calculate.

        rmax specifies the maximum radius to consider in the
        image. This is specified in coordinates of the original image
        that was fitted to.

        truncate specifies the number of basis function sigmas we
        consider either side of each point when calculating the
        intensity at each point. For example if truncate is 5.0, at
        each point we'll consider all basis functions whose centre
        lies within 5.0 * sigma of that point. 5.0 is the default.

        nthreads specifies the number of threads to use. If None, then
        we'll use all available cores.

        weighting specifies the weighting given to each pixel. If
        'normal', then no additional weighting is applied. If
        weighting='rsquared' then each pixel's value is weighted by
        the value of r squared for that pixel. If weighting='compund',
        then the negative x half of the image is weighted with r
        squared, and the +x half of the image is weighted
        normally. For 'normal' and 'rsquared' the image is normalized
        to a maximum value of 1.0. For 'compound' each half of the
        image is normalized to a maximum value of 1.0.

        """
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError

        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError

        # Set enpoint=False here, since the x, y values are the lowest
        # value of x, y in each bin.
        xvals = numpy.linspace(-rmax, rmax, bins, endpoint=False)
        yvals = numpy.linspace(-rmax, rmax, bins, endpoint=False)

        xbinw = xvals[1] - xvals[0]
        ybinw = yvals[1] - yvals[0]

        dist = numpy.zeros((bins, bins))
        queue = Queue.Queue(0)

        # Here we exploit the mirror symmetry in the y axis
        for xbin in numpy.arange(bins / 2, bins):
            xval = xvals[xbin] + 0.5 * xbinw # value at centre of pixel
            xval2 = xval * xval
            for ybin in numpy.arange(bins):
                yval = yvals[ybin] + 0.5 * ybinw # value at centre of pixel
                yval2 = yval * yval
                if math.sqrt(xval2 + yval2) <= self.rmax:
                    queue.put(
                        {'xbin': xbin, 'ybin': ybin, 'xval': xval, 'yval': yval})

        if self.oddl:
            oddl = 1
        else:
            oddl = 0

        def __worker():
            while not queue.empty():
                job = queue.get()
                xbin = job['xbin']
                ybin = job['ybin']
                xval = job['xval']
                yval = job['yval']

                #logger.debug('Calculating cartesian distribution at x={0}, y={1}'.format(xvals[xbin], yvals[ybin]))
                dist[xbin, ybin] = cartesian_distribution_point(
                    xval, yval, self.coef, self.kmax, self.rkstep,
                    self.sigma, self.lmax, oddl, truncate)
                #logger.debug('Finished calculating cartesian distribution at x={0}, y={1}'.format(xvals[xbin], yvals[ybin]))
                queue.task_done()

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        for i in range(nthreads):
            t = threading.Thread(target=__worker)
            t.daemon = True
            t.start()

        queue.join()

        # Mirror symmetry
        if bins % 2 != 0:  # bins is odd
            dist[bins / 2 - 1::-1] = dist[bins / 2 + 1:bins]
        else:  # bins is even
            dist[bins / 2 - 1::-1] = dist[bins / 2:bins]

        # Normalize image to max value of 1
        dist /= dist.max()

        # Now we weight with r squared if requested.
        if (weighting == 'rsquared') or (weighting == 'compound'):
            # r^2 weighting - we need the value of r at each pixel
            xm, ym = numpy.meshgrid(xvals, yvals)
            xm += 0.5 * xbinw
            ym += 0.5 * ybinw
            rsq = xm * xm + ym * ym
            if weighting == 'compound':
                if bins % 2 != 0:  # bins is odd
                    dist[bins / 2 - 1::-1] *= rsq[bins / 2 - 1::-1]
                    dist[bins / 2 - 1::-1] /= dist[bins / 2 - 1::-1].max()
                else:  # bins is even
                    dist[bins / 2 - 1::-1] *= rsq[bins / 2 - 1::-1]
                    dist[bins / 2 - 1::-1] /= dist[bins / 2 - 1::-1].max()
            else:
                dist *= rsq
                dist /= dist.max()

        return vmi.CartesianImage(x=xvals, y=yvals, image=dist)

    def cosn_expval(self, nmax=None, rbins=500, rmax=None,
                    truncate=5.0, nthreads=None,
                    epsabs=1.0e-7, epsrel=1.0e-7):
        '''Calculates the expectation values of cos^n(theta) for the fit as a
        function r up to rmax and for n from 0 to nmax.

        rbins specifies the number of data points to calculate.

        rmax specifies the maximum radius to consider and is specified
        in dimensions of the original image that was fitted.

        nthreads specifies the number of threads to be used. If None,
        then the number of CPU cores is used as the number of threads.

        truncate specifies the number of basis function sigmas we
        consider either side of each point when calculating the
        intensity at each point. For example if truncate is 5.0, at
        each point we'll consider all basis functions whose centre
        lies within 5.0 * sigma of that point. 5.0 is the default.

        epsabs and epsrel specify the absolute and relative error
        desired when performing the numerical integration over theta
        when calculating the expectatino values. The default values
        should suffice.

        '''
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError

        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError

        if nmax is None:
            nmax = self.lmax
        elif nmax > self.lmax:
            logger.error('nmax exceeds lmax of the fit')
            raise ValueError

        if self.oddl:
            oddl = 1
        else:
            oddl = 0

        # Calculate r values. Set enpoint=False here, since the r
        # values are the lowest value of r in each bin.
        r = numpy.linspace(0.0, rmax, rbins, endpoint=False)

        expval = numpy.zeros((nmax + 1, rbins))
        queue = Queue.Queue(0)

        for rbin in xrange(rbins):
            queue.put({'rbin': rbin})

        def __worker():
            while not queue.empty():
                job = queue.get()
                rbin = job['rbin']
                rr = r[rbin]

                expval[:, rbin] = cosn_expval_point(
                    rr, nmax, self.coef, self.kmax, self.rkstep,
                    self.sigma, self.lmax, oddl, truncate,
                    epsabs, epsrel)
                queue.task_done()

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        for i in xrange(nthreads):
            t = threading.Thread(target=__worker)
            t.daemon = True
            t.start()

        queue.join()

        return r, expval

    def beta_coefficients_threaded(self, rbins=500, rmax=None,
                                   truncate=5.0, nthreads=None):
        '''Calculates the beta coefficients for the fit as a function of
        r up to rmax and for n from 0 to nmax.

        rbins specifies the number of data points to calculate.

        rmax specifies the maximum radius to consider and is specified
        in dimensions of the original image that was fitted.

        nthreads specifies the number of threads to be used. If None,
        then the number of CPU cores is used as the number of threads.

        truncate specifies the number of basis function sigmas we
        consider either side of each point when calculating the
        intensity at each point. For example if truncate is 5.0, at
        each point we'll consider all basis functions whose centre
        lies within 5.0 * sigma of that point. 5.0 is the default.

        '''
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError

        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError

        if self.oddl:
            oddl = 1
        else:
            oddl = 0

        # Calculate r values. Set enpoint=False here, since the r
        # values are the lowest value of r in each bin.
        r = numpy.linspace(0.0, rmax, rbins, endpoint=False)

        beta = numpy.zeros((self.lmax + 1, rbins))
        queue = Queue.Queue(0)

        for rbin in xrange(rbins):
            queue.put({'rbin': rbin})

        def __worker():
            while not queue.empty():
                job = queue.get()
                rbin = job['rbin']
                rr = r[rbin]

                beta[:, rbin] = beta_coeffs_point(
                    rr, self.coef, self.kmax, self.rkstep,
                    self.sigma, self.lmax, oddl, truncate)
                queue.task_done()

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        for i in xrange(nthreads):
            t = threading.Thread(target=__worker)
            t.daemon = True
            t.start()

        queue.join()

        return r, beta

    def beta_coefficients(self, rbins=500, rmax=None):
        '''Calculates the beta coefficients for the fit as a function
        of r up to rmax.

        rbins specifies the number of data points calculated

        rmax specifies the maximum radius to consider and is specified
        in dimensions of the original image that was fitted.
        '''
        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError

        if rmax == None:
            rmax = self.rmax

        beta = beta_coeffs(rmax, rbins, self.coef, self.kmax,
                           self.rkstep, self.sigma, self.lmax)

        # Calculate r values. Set enpoint=False here, since the r
        # values are the lowest value of r in each bin.
        r = numpy.linspace(0.0, rmax, rbins, endpoint=False)

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
