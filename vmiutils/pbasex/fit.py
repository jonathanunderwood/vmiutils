# Copyright (C) 2014 by Jonathan G. Underwood.
#
# This file is part of VMIUtils.
#
# VMIUtils is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# VMIUtils is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with VMIUtils.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy.linalg
import cPickle as pickle
import Queue
import threading
import multiprocessing
import math
import concurrent.futures as futures
import matplotlib
import scipy.linalg
import warnings

import vmiutils as vmi
import matrix as pbm
import vmiutils.landweber
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


def _odd(n):
    if n % 2:
        return True
    else:
        return False


def _even(n):
    if n % 2:
        return False
    else:
        return True


class PbasexFit(object):

    # This private attribute is a list containing the variables that should be
    # saved to a file when dump is called and read when load is called.
    _metadata = ['kmax', 'lmax', 'oddl', 'sigma', 'rkstep', 'rmax']

    def __init__(self):
        self.coef = None
        self.kmax = None
        self.lmax = None
        self.oddl = None
        self.sigma = None
        self.rkstep = None
        self.rmax = None
        self.vmi_image = None

    def _build_CartesianImage(self, image, x, y, centre, swapxy):
        # Wrap VMI data into instance of CartesianImage
        if swapxy is True:
            image = image.transpose()
            if x is not None and y is not None:
                newx = y
                newy = x
                x = newx
                y = newy
                if centre is not None:
                    centre = (centre[1], centre[0])
                else:
                    centre = None
        else:
            if centre is not None:
                centre = (centre[0], centre[1])
            else:
                centre = None

        cart = vmi.CartesianImage(image=image, x=x, y=y)

        # Set centre
        if centre is not None:
            cart.set_centre(centre)

        return cart

    def fit(self, image, matrix, x=None, y=None, centre=None,
            swapxy=False, section='whole', lmax=None,
            oddl=None, Rmin=None, method='least_squares', cond=None,
            max_iterations=500, tolerance=1.0e-4):

        image_cart = self._build_CartesianImage(image, x, y, centre, swapxy)
        image_polar = vmi.PolarImage()
        image_polar.from_CartesianImage(image_cart, rbins=matrix.Rbins,
                                        thetabins=matrix.Thetabins)

        # Rmin as passed is a minimum radius in units of pixels of the
        # image. So, we need to scale it to pixels in the polar image
        # once it has been constructed using Rbins from matrix.
        if Rmin is None:
            Rbinmin = None
        else:
            Rvals = image_polar.r
            Rwidth = Rvals[1] - Rvals[0]
            Rbinmin = int(math.ceil(Rmin / float(Rwidth)))

        self.fit_data(image_polar, matrix, oddl=oddl, lmax=lmax,
                      Rbinmin=Rbinmin, method=method, cond=cond,
                      tolerance=tolerance, max_iterations=max_iterations)

        self.vmi_image = image_cart.zoom_circle(self.rmax, pad=True)

    def fit_data(self, image, matrix, section='whole', lmax=None, oddl=None,
                 Rbinmin=None, method='least_squares', cond=None,
                 max_iterations=500, tolerance=1.0e-4):
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

        Rbinmin specifies the minimum radial bin index used in the
        fit. Radial bins with indices less than Rbinmin are
        ignored. If this is None (the default), the the whole range of
        radial bins is used in the fit.

        method specifies the fitting method to use. Currently this can
        be 'least_squares', 'landweber', or 'projected_landweber'

        if method is 'least_squares' the following options are also used:

        cond: cutoff for small singular values; used to determine
        effective rank of the matrix. Singular values smaller than
        cond * largest_singular_value are considered zero. This allows
        some level of regularization to be achieved. Default is None
        (ignored).

        If method is 'landweber' or 'projected_landweber' the
        following options are also used:

        tolerance: the value used to terminate Landweber
        iteration. Iteration is terminated using a simple discrepancy
        calculation. The simulated vector b_k is calculated from the
        current x vector, x_k, from Ax_k=b_k. The norm of the
        discrepancy vector (i.e. b-b_k) is then calculated and
        normalized to the norm of b. If this value is less than
        tolerance, iteration is terminated.

        max_iterations: specifies the maximum number of iterations
        allowed before returning from Landweber iteration.

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

        # Decide whether to fit the whole range of Theta, or just
        # one half, and adjust image and matrix accordingly
        if section == 'whole':
            # Fit the whole image
            Thetadim = Thetabins
            img = image.image
        elif section == 'negative':
            # Fit only the part of the image in the region Theta = -Pi..0
            if _odd(Thetabins):
                endTheta = Thetabins / 2
            else:
                endTheta = (Thetabins / 2) - 1
            halfThetabins = endTheta + 1
            mtx = mtx[:, :, :, 0:endTheta]
            img = image.image[:, 0:endTheta]
            Thetadim = halfThetabins
        elif section == 'positive':
            # Fit only the part of the image in the region Theta = 0..Pi
            # Correct for both even and odd Thetabins
            startTheta = Thetabins / 2
            endtheta = Thetabins - 1
            halfThetabins = Thetabins - startTheta
            mtx = mtx[:, :, :, startTheta:endTheta]
            img = image.image[:, startTheta:endTheta]
            Thetadim = halfThetabins
        else:
            raise NotImplementedError

        kmax = matrix.kmax

        # If Rbinmin is specified, we don't want to fit bins less than
        # Rbinmin, so we remove them from the image and matrix
        if Rbinmin is None:
            Rdim = Rbins
            mtx = mtx.reshape((kdim * ldim, Rdim * Thetadim))
            img = img.reshape(Rdim * Thetadim)
            kmin = 0
        else:
            Rdim = Rbins - Rbinmin
            img = img[Rbinmin:, :]
            rkspacing = float(Rbins) / kmax
            kmin = int(math.floor(Rbinmin / rkspacing))
            mtx = mtx[kmin:, :, Rbinmin:, :]
            mtx = mtx.reshape(((kmax - kmin + 1) * ldim, Rdim * Thetadim))
            img = img.reshape(Rdim * Thetadim)

        # Now do the fitting according to the selected method.
        if method == 'least_squares':
            logger.debug('fitting with least squares')
            coef, resid, rank, s = scipy.linalg.lstsq(mtx.transpose(), img, cond=cond)
            # TODO: do something with resid
        elif method == 'landweber':
            logger.debug('fitting with Landweber iteration')

            coef = vmiutils.landweber.projected_landweber(mtx.T,
                                                          img,
                                                          max_iterations=max_iterations,
                                                          tolerance=tolerance,
                                                          filter_func=None,
                                                          )
        elif method == 'projected_landweber':
            logger.debug('fitting with projected Landweber iteration')
            krange = xrange(kmax - kmin + 1)

            def __filter(x, kmin, kmax, ldim, krange):
                c = x.reshape((kmax - kmin + 1, ldim))
                for k in krange:
                    if c[k, 0] < 0.0:
                        c[k, :] = 0.0

            coef = vmiutils.landweber.projected_landweber(mtx.T,
                                                          img,
                                                          max_iterations=max_iterations,
                                                          tolerance=tolerance,
                                                          filter_func=__filter,
                                                          extra_args=(kmin,
                                                                      kmax,
                                                                      ldim,
                                                                      krange), )
        else:
            raise NotImplementedError

        if Rbinmin is None:
            coef = coef.reshape((kdim, ldim))
        else:
            # Rebuild coefficient array with coefficients for K<kmin
            # equal to zero.
            c = coef.reshape((kmax - kmin + 1, ldim))
            coef = numpy.zeros((kdim, ldim))
            coef[kmin:, :] = c

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
        # Gaussian radial basis functions. Note that the division here
        # by kmax and not kmax+1 is consistent with matrix.py and
        # ensures that the last basis function is centred on rmax
        self.rkstep = rbinw * Rbins / float(matrix.kmax)

    def calc_radial_spectrum(self, rbins=500, rmax=None):
        msg = 'calc_radial_spectrum method is deprecated, use radial_spectrum method instead'
        logger.warning(msg)
        warnings.warn(msg, DeprecationWarning)
        return self.radial_spectrum(rbins=rbins, rmax=rmax)

    def radial_spectrum(self, rbins=500, rmax=None, truncate=5.0,
                        nthreads=None):
        """Calculate a radial spectrum from the parameters of a fit.  Returns
        a tuple (r, intensity) containing the r values and the
        corresponding intensities.

        rbins determines the number of points in the returned spectrum.

        rmax is the maximum radius to consider, i.e. the spectrum is
        calculated for r=0..rmax. Note: rmax is the desired radial
        value and not the bin number. If rmax is None (default) then
        the maximum radius in the input image is used.

        truncate specifies the number of basis function sigmas we
        consider either side of each point when calculating the
        intensity at each point. For example if truncate is 5.0, at
        each point we'll consider all basis functions whose centre
        lies within 5.0 * sigma of that point. 5.0 is the default.

        nthreads specifies the number of threads to use. If None, then
        we'll use all available cores.

        """

        if self.coef is None:
            logger.error('no fit done')
            raise AttributeError

        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError

        # Calculate r values. Set enpoint=False here, since the r
        # values are the lowest value of r in each bin.
        r = numpy.linspace(0.0, rmax, rbins, endpoint=False)

        spec = numpy.zeros(rbins)

        def __worker(rbin):
            rr = r[rbin]

            spec[rbin] = radial_spectrum_point(
                rr, self.coef, self.kmax, self.rkstep,
                self.sigma, self.lmax, truncate)

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        with futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
            jobs = dict((executor.submit(__worker, rbin), rbin)
                        for rbin in xrange(rbins))

            jobs_done = futures.as_completed(jobs)

            while True:
                try:
                    job = next(jobs_done)

                    if job.exception() is not None:
                        logger.error(job.exception())
                        for j in jobs:
                            if not j.done():
                                j.cancel()
                        raise job.exception()
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    logger.info('Ctrl-c received, exiting.')
                    for j in jobs:
                        if not j.done():
                            j.cancel()
                    raise
        # Normalize to max value of 1
        spec /= spec.max()

        return r, spec

    def cartesian_distribution_threaded(self, bins=250, rmax=None,
                                        truncate=5.0, nthreads=None,
                                        weighting='normal'):
        msg = 'cartesian_distribution_threaded method is deprecated, use cartesian_distribution method instead'
        logger.warning(msg)
        warnings.warn(msg, DeprecationWarning)
        return self.cartesian_distribution(bins=bins, rmax=rmax,
                                           truncate=truncate,
                                           nthreads=nthreads,
                                           weighting=weighting)

    def cartesian_distribution(self, bins=250, rmax=None,
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

        # Here we exploit the mirror symmetry in the y axis if bins is
        # an even number, since in this case the points are
        # symmetrically distributed about 0, unlike the case of odd
        # bins.

        if _even(bins):
            evenbins = True
            xstart = bins / 2
        else:
            evenbins = False
            xstart = 0

        for xbin in numpy.arange(xstart, bins):
            xval = xvals[xbin] + 0.5 * xbinw  # value at centre of pixel
            xval2 = xval * xval
            for ybin in numpy.arange(bins):
                yval = yvals[ybin] + 0.5 * ybinw  # value at centre of pixel
                yval2 = yval * yval
                if math.sqrt(xval2 + yval2) <= rmax:
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

                dist[xbin, ybin] = cartesian_distribution_point(
                    xval, yval, self.coef, self.kmax, self.rkstep,
                    self.sigma, self.lmax, oddl, truncate)

                queue.task_done()

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        for i in range(nthreads):
            t = threading.Thread(target=__worker)
            t.daemon = True
            t.start()

        queue.join()

        # Mirror symmetry
        if evenbins:
            dist[xstart - 1::-1] = dist[xstart:bins]

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
                # Note that this won't be quite a symmetrical image
                # for the case that bins is an odd number.
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

        def __worker(rbin):
            rr = r[rbin]

            expval[:, rbin] = cosn_expval_point(
                rr, nmax, self.coef, self.kmax, self.rkstep,
                self.sigma, self.lmax, oddl, truncate,
                epsabs, epsrel)
            # import time
            # time.sleep(1)

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        with futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
            jobs = dict((executor.submit(__worker, rbin), rbin)
                        for rbin in xrange(rbins))

            jobs_done = futures.as_completed(jobs)

            while True:
                try:
                    job = next(jobs_done)

                    if job.exception() is not None:
                        logger.error(job.exception())
                        for j in jobs:
                            if not j.done():  # and not j.running():
                                j.cancel()
                        raise job.exception()
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    logger.info('Ctrl-c received, exiting.')
                    for j in jobs:
                        if not j.done():  # and not j.running():
                            j.cancel()
                    raise

        return r, expval

    def cosn_expval2(self, nmax=None, rbins=500, rmax=None,
                     truncate=5.0, nthreads=None,
                     epsabs=1.0e-7, epsrel=1.0e-7):
        msg = 'cosn_expval2 method is deprecated, use cosn_expval method instead'
        logger.warning(msg)
        warnings.warn(msg, DeprecationWarning)
        return self.cosn_expval(nmax=nmax, rbins=rbins, rmax=rmax,
                                truncate=truncate, nthreads=nthreads,
                                epsabs=epsabs, epsrel=epsrel)

    def beta_coefficients(self, rbins=500, rmax=None,
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

    def beta_coefficients_threaded(self, rbins=500, rmax=None,
                                   truncate=5.0, nthreads=None):
        msg = 'beta_coefficients_threaded method is deprecated, use beta_coefficients method instead'
        logger.warning(msg)
        warnings.warn(msg, DeprecationWarning)

        return self.beta_coefficients(rbins=rbins, rmax=rmax,
                                      truncate=truncate,
                                      nthreads=nthreads)

    def dumpfd(self, fd):
        for object in self._metadata:
            pickle.dump(getattr(self, object), fd, protocol=2)
        numpy.save(fd, self.coef)
        self.vmi_image.dump(fd)

    def loadfd(self, fd):
        for object in self._metadata:
            setattr(self, object, pickle.load(fd))
        self.coef = numpy.load(fd)
        self.vmi_image = vmi.CartesianImage()
        self.vmi_image.load(fd)

    def dump(self, file):
        fd = open(file, 'w')
        self.dumpfd(fd)
        fd.close()

    def load(self, file):
        fd = open(file, 'r')
        try:
            self.loadfd(fd)
        finally:
            fd.close()

class PbasexFitVMI(object):
    def __init__(self, fit):
        self.vmi_image = fit.vmi_image

    def plot(self, axis, cmap=matplotlib.cm.spectral,
             xlabel=None, ylabel=None, rasterized=True,
             transpose=False, clip=None, origin_at_centre=True):

        im = self.vmi_image.plot(axis, cmap=cmap,
                                 rasterized=rasterized,
                                 transpose=transpose,
                                 clip=clip,
                                 origin_at_centre=origin_at_centre)
        if xlabel is None:
            if transpose is False:
                axis.set_xlabel(r'$y$ (pixels)')
            else:
                axis.set_xlabel(r'$z$ (pixels)')
        else:
            axis.set_xlabel(xlabel)

        if ylabel is None:
            if transpose is False:
                axis.set_ylabel(r'$z$ (pixels)')
            else:
                axis.set_ylabel(r'$y$ (pixels)')
        else:
            axis.set_ylabel(ylabel)

        axis.axis('image')

        return im

def _augment(arr):
    return numpy.append(arr, arr[-1] + arr[1] - arr[0])


class PbasexFitCartesianImage(object):

    def __init__(self, fit, bins=500):
        self.image = fit.cartesian_distribution(bins=bins)

    def plot(self, axis, cmap=matplotlib.cm.spectral,
             xlabel=None, ylabel=None, rasterized=True,
             transpose=False, plot_type='image', clip=None):
        if transpose is False:
            x = self.image.x
            y = self.image.y
            image = self.image.image.T
        elif transpose is True:
            x = self.image.y
            y = self.image.x
            image = self.image.image
        else:
            raise ValueError('transpose must be True or False')

        if clip is not None:
            image = image.clip(clip)

        if plot_type is 'image':
            im = axis.pcolormesh(_augment(x), _augment(y),
                                 image, cmap=cmap,
                                 rasterized=rasterized)
        elif plot_type is 'contour':
            im = axis.contour(self.image.x, self.image.y,
                              self.image.image.T, origin='lower',
                              cmap=cmap, linewidths=0.5)
        else:
            raise NotImplementedError

        if xlabel is None:
            if transpose is False:
                axis.set_xlabel(r'$y$ (pixels)')
            else:
                axis.set_xlabel(r'$z$ (pixels)')
        else:
            axis.set_xlabel(xlabel)

        if ylabel is None:
            if transpose is False:
                axis.set_ylabel(r'$z$ (pixels)')
            else:
                axis.set_ylabel(r'$y$ (pixels)')
        else:
            axis.set_ylabel(ylabel)

        axis.axis('image')

        return im

class PbasexFitRadialSpectrum(object):
    xlabel = r'$r$ (pixels)'
    ylabel = r'$I$ (a.u)'

    def __init__(self, fit, rbins=500):
        self.r, self.spec = fit.calc_radial_spectrum(rbins=rbins)

    def plot(self, axis, linestyle='-',
             scale_min=None, scale_max=None,
             xlabel=None, ylabel=None,
             label=None, color=None):

        if color is not None:
            line = axis.plot(self.r, self.spec, linestyle=linestyle,
                             label=label, color=color)
        else:
            line = axis.plot(self.r, self.spec, linestyle=linestyle,
                             label=label)

        if xlabel is None:
            axis.set_xlabel(self.xlabel)
        else:
            axis.set_xlabel(xlabel)

        if ylabel is None:
            axis.set_ylabel(self.ylabel)
        else:
            axis.set_ylabel(ylabel)

        axis.set_xlim(self.r.min(), self.r.max())

        if scale_min is None:
            ymin = self.spec.min()
        else:
            ymin = scale_min

        if scale_max is None:
            ymax = self.spec.max()
        else:
            ymax = scale_max

        axis.set_ylim(ymin, ymax)

        return line

class PbasexFitBetaSpectrum(object):
    def __init__(self, fit, rbins=500):
        self.r, self.beta = fit.beta_coefficients_threaded(rbins=rbins)
        self.lmax = fit.lmax
        self.oddl = fit.oddl

    def plot(self, axis, betavals=None, rbins=500, scale_min=None,
             scale_max=None, xlabel=None, ylabel=None, cmap=matplotlib.cm.jet,
             linestyle='-', scaley=True):
        """Generate a plot of the r-dependent beta values.

        betavals is a list specifying which betavalues to plot. If
        this is None (the default), all beta values are plotted.

        rbins specifies the number of bins in r to use when generating
        the plot. Default is 500.

        scale_min specifies the minimum y-axis value. If this is None
        (the default), the smallest value of all the beta values is
        used.

        scale_max specifies the maximum y-axis value. If this is None
        (the default), the largest value of all the beta values is
        used.

        xlabel specifies the x-axis label. If this is None (the
        default), the 'r (pixels)' is used.

        ylabel specifies the y-axis label. If this is None (the
        default) and only one beta value is specified in betavals,
        then this will be 'beta_n' where n is the value of l for the
        plotted beta value. If more than one beta value is plotted,
        and ylabel is None, then 'beta_l' is used.

        scaley IS UNUSED AND WILL BE REMOVED IN THE NEAR FUTURE.

        """
        if scale_min is None:
            find_ymin = True
            ymin = None
        else:
            find_ymin = False
            ymin = scale_min

        if scale_max is None:
            find_ymax = True
            ymax = None
        else:
            find_ymax = False
            ymax = scale_max

        if betavals is None:
            if self.oddl is True:
                linc = 1
            else:
                linc = 2

            betavals = range(linc, self.lmax + 1, linc)

        if len(betavals) == 1:
            b = betavals[0]
            lines = axis.plot(self.r, self.beta[b],
                              label=r'$l=${0}'.format(b))
            if find_ymin is True:
                ymin = self.beta[b].min()

            if find_ymax is True:
                ymax = self.beta[b].max()

            if ylabel is None:
                axis.set_ylabel(r'$\beta_{{{0}}}$'.format(b))
            else:
                axis.set_ylabel(ylabel)

            if xlabel is None:
                axis.set_xlabel(r'$r$ (pixels)')
            else:
                axis.set_xlabel(xlabel)
        else:
            lines = []
            colors = iter(cmap(numpy.linspace(0.0, 0.95, len(betavals))))
            for b in betavals:
                color = next(colors)
                line = axis.plot(self.r, self.beta[b],
                                 label=r'$l=${0}'.format(b), color=color,
                                 linestyle=linestyle)
                lines.append(line)

                if find_ymin is True:
                    if ymin is None:
                        ymin = self.beta[b].min()
                    else:
                        ymin = min(ymin, self.beta[b].min())

                if find_ymax is True:
                    if ymax is None:
                        ymax = self.beta[b].max()
                    else:
                        ymax = max(ymax, self.beta[b].max())

            # TODO: add options for legend.
            # ax.legend(loc='upper left',
            #           bbox_to_anchor=(1.1, 1.0),
            #           fontsize=8)
            if ylabel is None:
                axis.set_ylabel(r'$\beta_l$')
            else:
                axis.set_ylabel(ylabel)

            if xlabel is None:
                axis.set_xlabel(r'$r$ (pixels)')
            else:
                axis.set_xlabel(xlabel)

        axis.set_autoscale_on(False)
        axis.set_ylim(ymin, ymax)
        axis.set_xlim(self.r.min(), self.r.max())

        return lines

class PbasexFitCosnSpectrum(object):
    def __init__(self, fit, rbins=500):
        self.r, self.cosn = fit.cosn_expval2(rbins=rbins)
        self.lmax = fit.lmax
        self.oddl = fit.oddl

    def plot(self, axis, nvals, rbins=500, scale_min=None,
             scale_max=None, xlabel=None, ylabel=None, cmap=matplotlib.cm.jet):
        ymin = scale_min
        ymax = scale_max

        if nvals is None:
            if self.oddl is True:
                linc = 1
            else:
                linc = 2

            nvals = range(linc, self.lmax + 1, linc)

        if len(nvals) == 1:
            n = nvals[0]

            lines = axis.plot(self.r, self.cosn[n],
                              label=r'$n=${0}'.format(n))
            if ylabel is None:
                axis.set_ylabel(r'$\langle\cos^{{{0}}}\theta\rangle$'.format(n))
            else:
                axis.set_ylabel(ylabel)

            if xlabel is None:
                axis.set_xlabel(r'$r$ (pixels)')
            else:
                axis.set_xlabel(xlabel)
        else:
            lines = []
            colors = iter(cmap(numpy.linspace(0.0, 0.95, len(nvals))))
            for n in nvals:
                color = next(colors)
                line = axis.plot(self.r, self.cosn[n],
                                 label=r'$n=${0}'.format(n), color=color)
                lines.append(line)

            if scale_min is None:
                if ymin is None:
                    ymin = cosn[n].min()
                else:
                    ymin = min(ymin, cosn[n].min())

            if scale_max is None:
                if ymax is None:
                    ymax = cosn[n].max()
                else:
                    ymax = max(ymax, cosn[n].max())

            # TODO: add options for legend.
            # ax.legend(loc='upper left',
            #           bbox_to_anchor=(1.1, 1.0),
            #           fontsize=8)
            if ylabel is None:
                axis.set_ylabel(r'$\langle\cos^n\theta\rangle$')
            else:
                axis.set_ylabel(ylabel)

            if xlabel is None:
                axis.set_xlabel(r'$r$ (pixels)')
            else:
                axis.set_xlabel(xlabel)

        axis.set_autoscale_on(False)
        axis.set_ylim(ymin, ymax)
        axis.set_xlim(self.r.min(), self.r.max())

        return lines
