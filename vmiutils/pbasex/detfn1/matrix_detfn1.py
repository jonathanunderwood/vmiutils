import numpy
import cPickle as pickle
import math as m
import logging
import multiprocessing
import futures

import vmiutils.pbasex as pbasex

from _matrix_detfn1 import *

# Set up logging and create a null handler in case the application doesn't
# provide a log handler
logger = logging.getLogger('vmiutils.pbasex.matrix_detfn1')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


class PbasexMatrixDetFn1 (pbasex.PbasexMatrix):

    def __init__(self):
        super(PbasexMatrixDetFn1, self).__init__()
        self.rmax = None
        self.description = 'pbasex_detfn1_matrix'
        self._metadata += ['method',
                           'threshold',
                           'rmax']

    def calc_matrix_threaded(self, Rbins, Thetabins, kmax, lmax,
                             detectionfn, alpha=0.0, beta=0.0,
                             sigma=None, oddl=True, method='cquad',
                             epsabs=0.0, epsrel=1.0e-7, threshold=1.0e-18,
                             wkspsize=100000, nthreads=None):
        """Calculates an inversion matrix using multiple threads.

        kmax determines the number of radial basis functions (from
        k=0..kmax).

        lmax determines the maximum value of l for the legendre
        polynomials (l=0..lmax).

        Rbins specifies the number of radial bins in the image to be
        inverted.

        Thetabins specifies the number of angular bins in the image to
        be inverted.

        sigma specifes the width of the Gaussian radial basis
        functions. This is defined according to the normal convention
        for Gaussian functions i.e. FWHM=2*sigma*sqrt(2*ln2), and NOT
        as defined in the Garcia, Lahon, Powis paper. If sigma is not
        specified it is set automatically such that the half-maximum
        of the Gaussian occurs midway between each radial
        function. Note that sigma is specified indimension of bins of
        the image, NOT in the dimensions of the detection function.

        method specifies the integration method to be used. Currently
        supported values are 'qaws', 'qags' and 'cquad' corresponding
        to the different GSL integration functions of the same name.

        epsabs and epsrel specify the desired integration tolerance
        when calculating the basis functions. The defaults should
        suffice.

        wkspsize specifies the maximum number of subintervals used for
        the numerical integration of the basis functions.

        threshold specifies the lowest value of the integrand which will
        not be set to zero during the basis function calculation at
        each point. I.e. any value of the intgrand below thresh will
        be set to zero. This is necessary because numerical noise in
        the integrand can prevent the integration routine from
        converging. A value of 1.0e-18 is probably sufficient, but it
        may need to be increased if the detection function is
        particularly badly behaved.

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
        if not isinstance(detectionfn, pbasex.PbasexFit):
            raise TypeError('detectionfn is not an instance of PbasexFit')

        # Spacing of radial basis function centres. The most obvious
        # choice here is rkspacing = Rbins / (kmax + 1.0), but we'd
        # actually like to have the last basis function centered on
        # the largest value of R, so instead we choose:
        rkspacing = Rbins / float(kmax)

        if sigma is None:
            # If sigma is not specified, we calculate a reasonable
            # value based on rkspacing. In the original Powis et al
            # paper they had rkspacing=2, and set their sigma (=2
            # sigma^2) = 2 pixels. We do similar here, and then make
            # it 20% bigger to avoid oscillations.
            sigma = m.sqrt(rkspacing / 2.0) * 1.2

        # We need to rescale the detection function parameters to
        # express them in terms of bins for the actual matrix
        # calculation. The detection function rmax specifies the
        # maximum radial value we can sensibly consider. The stored
        # values of rkstep and sigma in the detection function are
        # currently scaled according to that rmax (i.e. are in
        # dimensions of the image used to generate the detection
        # function fit). So, since our matrix calculation is actually
        # done in terms of bins, rather than absolute scale, we need
        # to rescale the detection function parameters accordingly.
        df_rscale = detectionfn.rmax / Rbins
        df_rkstep = detectionfn.rkstep / df_rscale
        df_sigma = detectionfn.sigma / df_rscale

        # Normalize detection function coefficients to max of 1
        detectionfn.coef /= detectionfn.coef.max()

        if detectionfn.oddl is True:
            df_oddl = 1
        else:
            df_oddl = 0

        if oddl is False:
            linc = 2
            mtx = numpy.empty([kmax + 1, lmax / 2 + 1, Rbins, Thetabins])
        else:
            linc = 1
            mtx = numpy.empty([kmax + 1, lmax + 1, Rbins, Thetabins])

        def __worker(k, l):
            rk = rkspacing * k

            logger.info(
                'Calculating basis function for k={0}, l={1}'.format(k, l))

            bf = basisfn_detfn1(
                k, l, Rbins, Thetabins, sigma, rk,
                epsabs, epsrel, wkspsize, threshold,
                detectionfn.coef, detectionfn.kmax, df_sigma,
                df_rkstep, detectionfn.lmax, df_oddl,
                alpha, beta, method)

            if oddl is True:
                mtx[k, l] = bf
            else:
                mtx[k, l / 2] = bf

            logger.info(
                'Finished calculating basis function for k={0}, l={1}'.format(
                    k, l)
            )

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        with futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
            jobs = []

            for k in numpy.arange(kmax + 1):
                for l in numpy.arange(0, lmax + 1, linc):
                    j = executor.submit(__worker, k, l)
                    jobs.append(j)

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

        self.matrix = mtx
        self.kmax = kmax
        self.sigma = sigma
        self.lmax = lmax
        self.oddl = oddl
        self.Rbins = Rbins
        self.Thetabins = Thetabins
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.method = method
        self.threshold = threshold
        # It's important we save this as part of the matrix
        # object, as subsequent fits with this matrix are only
        # valid if they have the same binning and scaling. Note
        # that we can always calculate the relevant rscale from
        # (self.Rbins / self.rmax) if needs be, so we don't save
        # that.
        self.rmax = detectionfn.rmax

if __name__ == "__main__":
    import vmiutils as vmi
    import pylab
    import matplotlib
    import matplotlib.pyplot as plot

    Rbins = 256
    Thetabins = 256
    kmax = 128
    rkspacing = Rbins / kmax
    sigma = rkspacing / (2.0 * m.sqrt(2.0 * m.log(2.0)))
    k = 32
    rk = k * rkspacing
    l = 2
    alpha = m.radians(0.0)
    beta = m.radians(90.0)
    epsabs = 0.0
    epsrel = 1.0e-7
    wkspsize = 100000
    method = 'cquad'
    threshold = 1.0e-18
    dffile = '/home/jgu/Code/vmi_invert/aarhus/junderwood/Probe.pbfit'
    detectionfn = pbasex.PbasexFit()
    detectionfn.load(dffile)

    if detectionfn.oddl is True:
        df_oddl = 1
    else:
        df_oddl = 0

    bf = basisfn_detfn1(
        k, l, Rbins, Thetabins, sigma, rk,
        epsabs, epsrel, wkspsize, threshold,
        detectionfn.coef, detectionfn.kmax, detectionfn.sigma,
        detectionfn.rkstep, detectionfn.lmax, df_oddl,
        alpha, beta, method)

    r = numpy.linspace(0, Rbins + 1, Rbins, endpoint=False)
    theta = numpy.linspace(-numpy.pi, numpy.pi, Thetabins, endpoint=False)

    r_aug = numpy.append(r, r[-1] + r[1] - r[0])
    theta_aug = numpy.append(theta, theta[-1] + theta[1] - theta[0])
    # Plot using polar projection
    gs = matplotlib.gridspec.GridSpec(1, 2,
                                      width_ratios=[10,1],
                                  )

    ax = plot.subplot(gs[0], projection="polar", aspect=1.)
    cb = plot.subplot(gs[1])
    im = ax.pcolormesh(theta_aug, r_aug, bf)
    ax.grid()
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_title('Polar data\n(pcolormesh/polar\nprojection)')
    plot.colorbar(im, cax=cb)
    plot.show()
