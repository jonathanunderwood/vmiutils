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

import vmiutils as vmi
import vmiutils.pbasex as pbasex
import logging
import numpy
import numpy.polynomial.legendre as legendre
import Queue
import threading
import multiprocessing
import concurrent.futures as futures
import math
import scipy.optimize
from vmiutils.pbasex._fit import *
from _fit_detfn1 import *

logger = logging.getLogger('vmiutils.pbasex.fit')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

def _even(n):
    if n % 2:
        return False
    else:
        return True

class PbasexFitDetFn1(pbasex.PbasexFit):

    def __init__(self):
        super(PbasexFitDetFn1, self).__init__()
        self.detectionfn = None
        self.alpha = None
        self.beta = None
        self._metadata += ['alpha',
                           'beta']

    def fit(self, image, matrix, x=None, y=None, centre=None,
            swapxy=False, section='whole', lmax=None,
            oddl=None, Rmin=None, method='least_squares', cond=None,
            max_iterations=500, tolerance=1.0e-4):

        image_cart = self._build_CartesianImage(image, x, y, centre, swapxy)

        # This next part is key. The matrix has been calculated using a
        # previous pbasex fit, and so (unlike the general pbasex case) the
        # matrix is only meaningful for a polar image with the same radial
        # bin width and maximum radius.
        logger.debug('zooming into image to match matrix rmax={0}'.format(matrix.rmax))
        image_cart_zoom = image_cart.zoom_circle(matrix.rmax, pad=True)
        logger.debug('zoom of image created to match matrix rmax={0}'.format(matrix.rmax))

        image_polar = vmi.PolarImage()
        image_polar.from_CartesianImage(image_cart_zoom, rbins=matrix.Rbins,
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

        self.fit_data(image_polar, matrix, oddl=oddl, lmax=lmax, Rbinmin=Rbinmin,
                      method=method, cond=cond, tolerance=tolerance,
                      max_iterations=max_iterations)

        self.vmi_image = image_cart_zoom

        # Store the detection function as well, so that we can
        # calculate overlap functions of the extracted distribution
        # with the detection function.
        self.detectionfn = matrix.detectionfn
        self.alpha = matrix.alpha
        self.beta = matrix.beta

    def detectionfn_cartesian_distribution(self, bins=250, rmax=None,
                                               truncate=5.0, nthreads=None,
                                               weighting='normal', func=None):
        """Calculates a cartesian image of the detection function distribution
        integrated over phi in the y-z plane i.e. the plane that we normally
        plot the extracted axis distribution in. Multithreaded for speed.

        bins specifes the number of bins in the y and z dimension to
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
            rmax = self.detectionfn.rmax
        elif rmax > self.detectionfn.rmax:
            logger.error('rmax exceeds that of original detection function data')
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

        if self.detectionfn.oddl:
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
                dist[xbin, ybin] = detfn_cartesian_distribution_point(
                    xval, yval, self.beta, self.detectionfn.coef, 
                    self.detectionfn.kmax, self.detectionfn.rkstep,
                    self.detectionfn.sigma, self.detectionfn.lmax, 
                    oddl, truncate)
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


    def overlap_factor(self, rbins=500, rmax=None,
                       truncate=5.0, nthreads=None):
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

        df = self.detectionfn
        if df.oddl:
            df_oddl = 1
        else:
            df_oddl = 0

        # Calculate r values. Set enpoint=False here, since the r
        # values are the lowest value of r in each bin.
        r = numpy.linspace(0.0, rmax, rbins, endpoint=False)

        lmax = min(self.lmax, df.lmax)

        wt = numpy.fromfunction(lambda l: 1 / (2 * l + 1),
                               (lmax + 1,))

        # Construct array holding the legendre polynomials evaluated at self.beta
        plbeta = legendre.legval(math.cos(self.beta), numpy.diag(numpy.ones(lmax + 1)))

        overlap = numpy.zeros(rbins)

        def __worker(rbin):
            rr = r[rbin]

            # Axis distribution beta parameters
            beta_axis = beta_coeffs_point(
                rr, self.coef, self.kmax, self.rkstep,
                self.sigma, self.lmax, oddl, truncate)

            # Detection function beta parameters in the detection frame
            beta_df = beta_coeffs_point(
                rr, df.coef, df.kmax, df.rkstep,
                df.sigma, df.lmax, df_oddl, truncate)

            # Detection function beta parameters in the lab frame
            beta_df = beta_df[0:lmax + 1] * plbeta

            # The following is too slow
            # res = scipy.optimize.basinhopping(
            #     lambda x: -legendre.legval(x, beta_df),
            #     [0.0], stepsize = 1.0/lmax, T=1.0,
            #     minimizer_kwargs=dict(method='TNC', bounds=((-1.0,1.0),))
            #     )

            # Find approximate position of maxmimum. Note that there
            # are lmax maxima/minima between -1..1, so this value of
            # Ns should suffice.
            res = scipy.optimize.brute(
                lambda x, *args: -legendre.legval(x, beta_df),
                ((-1.0, 1.0),), Ns=5.0 * lmax,
            )

            # Refine position of maximum - note that the 'finish'
            # option of optimize.brute doesn't work with fmin_tnc etc
            respolish = scipy.optimize.minimize(lambda x: -legendre.legval(x, beta_df),
                                                [res[0]], bounds=((-1.0,1.0),), method='TNC')

            maxval = -respolish.fun

            beta = beta_axis[0:lmax + 1] * beta_df[0:lmax + 1] * wt

            overlap[rbin] = beta.sum() / maxval

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

        return r, overlap

    def probe_radial_spectrum(self, rbins=500):
        return self.detectionfn.calc_radial_spectrum(rbins=rbins)

    def dumpfd(self, fd):
        super(PbasexFitDetFn1, self).dumpfd(fd)
        self.detectionfn.dumpfd(fd)

    def loadfd(self, fd):
        super(PbasexFitDetFn1, self).loadfd(fd)
        self.detectionfn = pbasex.PbasexFit()
        self.detectionfn.loadfd(fd)

class PbasexFitDetFn1OverlapSpectrum(object):
    def __init__(self, fit, rbins=500):
        self.r, self.overlap = fit.overlap_factor(rbins=rbins)

    def plot(self, axis, linestyle='-', scale_min=None, scale_max=None,
             xlabel=None, ylabel=None):

        line = axis.plot(self.r, self.overlap, linestyle=linestyle)

        if xlabel is None:
            axis.set_xlabel(r'$r$ (pixels)')#, style='italic')
        else:
            axis.set_xlabel(xlabel)

        if ylabel is None:
            axis.set_ylabel(r'Overlap (a.u)')#, style='italic')
        else:
            axis.set_ylabel(ylabel)

        axis.set_autoscale_on(False)

        if scale_min is None:
            ymin = self.overlap.min()
        else:
            ymin = scale_min

        if scale_max is None:
            ymax = self.overlap.max()
        else:
            ymax = scale_max

        axis.set_ylim(ymin, ymax)
        axis.set_xlim(self.r.min(), self.r.max())

        return line

class PbasexFitDetFn1ProbeRadialSpectrum(pbasex.PbasexFitRadialSpectrum):
    xlabel = r'$r$ (pixels)'
    ylabel = r'$I_\mathrm{probe}$ (a.u)'

    def __init__(self, fit, rbins=500):
        self.r, self.spec = fit.probe_radial_spectrum(rbins=rbins)

class PbasexFitDetFn1DetectionFnCartesianDistribution(pbasex.PbasexFitCartesianImage):
    def __init__(self, fit, bins=500):
        self.image = fit.detectionfn_cartesian_distribution(bins=bins)

class PbasexFitDetFn1VMI(pbasex.PbasexFitVMI):
    pass

class PbasexFitDetFn1CartesianImage(pbasex.PbasexFitCartesianImage):
    pass

class PbasexFitDetFn1RadialSpectrum(pbasex.PbasexFitRadialSpectrum):
    pass

class PbasexFitDetFn1BetaSpectrum(pbasex.PbasexFitBetaSpectrum):
    pass

class PbasexFitDetFn1CosnSpectrum(pbasex.PbasexFitCosnSpectrum):
    pass
