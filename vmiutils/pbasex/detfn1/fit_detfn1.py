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
import multiprocessing
import futures
import math
import scipy.optimize
from vmiutils.pbasex._fit import *

logger = logging.getLogger('vmiutils.pbasex.fit')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

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
            oddl=None, method='least_squares',
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

        self.fit_data(image_polar, matrix, oddl=oddl, lmax=lmax, method=method,
                      tolerance=tolerance, max_iterations=max_iterations)

        self.vmi_image = image_cart_zoom

        # Store the detection function as well, so that we can
        # calculate overlap functions of the extracted distribution
        # with the detection function.
        self.detectionfn = matrix.detectionfn
        self.alpha = matrix.alpha
        self.beta = matrix.beta

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
            axis.set_ylabel(r'Angular overlap factor$(r)$ (a.u)')#, style='italic')
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
    ylabel = r'$I_\mathrm{probe}(r)$ (a.u)'

    def __init__(self, fit, rbins=500):
        self.r, self.spec = fit.probe_radial_spectrum(rbins=rbins)
