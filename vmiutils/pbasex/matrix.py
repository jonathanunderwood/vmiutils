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

import numpy
import cPickle as pickle
import math as m
import logging
import multiprocessing
import Queue
import threading

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


class PbasexMatrix(object):
    # This private attribute is a list containing the variables that should be
    # saved to a file when dump is called and read when load is called.
    _metadata = ['Rbins', 'Thetabins', 'kmax', 'sigma', 'lmax', 'oddl',
                 'epsabs', 'epsrel', 'description']

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
        self.description = 'pbasex_matrix'

    def calc_matrix_threaded(self, Rbins, Thetabins, kmax, lmax, sigma=None,
                             oddl=True, epsabs=0.0, epsrel=1.0e-7,
                             wkspsize=100000, nthreads=None):
        """Calculates an inversion matrix using multiple threads.

        kmax determines the number of radial basis functions (from k=0..kmax).

        lmax determines the maximum value of l for the legendre polynomials
        (l=0..lmax).

        Rbins specifies the number of radial bins in the image to be inverted.

        Thetabins specifies the number of angular bins in the image to be
        inverted.

        sigma specifes the width of the Gaussian radial basis
        functions. This is defined according to the normal convention
        for Gaussian functions i.e. FWHM=2*sigma*sqrt(2*ln2), and NOT
        as defined in the Garcia, Lahon, Powis paper. If sigma is not
        specified it is set automatically to 1.2*sqrt(rkspacing / 2.0)
        where rkspacing is the radial basis function spacing
        (Rbins/kmax). In other words, the default FWHM of the radial
        basis functions will be 2.4*sqrt[ln(2)*rkspacing].

        epsabs and epsrel specify the desired integration tolerance when
        calculating the basis functions. The defaults should suffice.

        tolerance specifies the acceptable relative error returned from the
        numerical integration. The default value should suffice.

        wkspsize specifies the maximum number of subintervals used for the
        numerical integration of the basis functions.

        nthreads specifies the number of threads to use. If this has
        the default value of None, the number of threads used will be
        equal to the number of CPU cores.

        """
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

        if oddl is False:
            linc = 2
            mtx = numpy.empty([kmax + 1, lmax / 2 + 1, Rbins, Thetabins])
        else:
            linc = 1
            mtx = numpy.empty([kmax + 1, lmax + 1, Rbins, Thetabins])

        queue = Queue.Queue(0)

        for k in xrange(kmax + 1):
            for l in xrange(0, lmax + 1, linc):
                queue.put({'k': k, 'l': l})

        shutdown_event = threading.Event()

        def __worker():
            while (not queue.empty()) and (not shutdown_event.is_set()):
                job = queue.get()
                k = job['k']
                l = job['l']

                rk = rkspacing * k

                logger.info(
                    'Calculating basis function for k={0}, l={1}'.format(k, l))

                try:
                    bf = basisfn(k, l, Rbins, Thetabins, sigma, rk,
                                 epsabs, epsrel, wkspsize)

                    if oddl is True:
                        mtx[k, l] = bf
                    else:
                        mtx[k, l / 2] = bf

                    logger.info(
                        'Finished calculating basis function for k={0}, l={1}'.format(k, l))
                    queue.task_done()
                except IntegrationError as errstring:
                    logger.info(errstring)
                    shutdown_event.set()

            logger.info('Exiting')

        if nthreads is None:
            nthreads = multiprocessing.cpu_count()

        threads = []
        for i in range(nthreads):
            t = threading.Thread(target=__worker)
            threads.append(t)
            t.daemon = True
            t.start()

        # Can't use queue.join() here as this will block forever if an
        # exception is raised in a thread and the threads are killed
        # before the queue is emptied. Instead we'll wait for the
        # threads to have all exited and then see if an error
        # occured. As a by-product, we can also trap Ctrl-C presses.
        while len(threads) > 0:
            try:
                # Join all threads using a timeout so it doesn't block
                # Filter out threads which have been joined or are None
                threads = [t.join(1000)
                           for t in threads if t is not None and t.isAlive()]
            except KeyboardInterrupt:
                logger.info('Ctrl-c received, exiting')
                shutdown_event.set()

        if shutdown_event.is_set() or not queue.empty():
            logger.info('Error calculating matrix')
            raise RuntimeError('Error calculating matrix')

        self.matrix = mtx
        self.kmax = kmax
        self.sigma = sigma
        self.lmax = lmax
        self.oddl = oddl
        self.Rbins = Rbins
        self.Thetabins = Thetabins
        self.epsabs = epsabs
        self.epsrel = epsrel

    def dumpfd(self, fd):
        for object in self._metadata:
            pickle.dump(getattr(self, object), fd, protocol=2)
        numpy.save(fd, self.matrix)

    def loadfd(self, fd):
        for object in self._metadata:
            setattr(self, object, pickle.load(fd))
        self.matrix = numpy.load(fd)

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

    def print_params(self):
        for object in self._metadata:
            print('{0}: {1}'.format(object, getattr(self, object)))


if __name__ == "__main__":
    import vmiutils as vmi
    import matplotlib.pyplot as plot
    import matplotlib.gridspec as gridspec

    Rbins = 256
    Thetabins = 256
    kmax = 128
    rkspacing = Rbins / kmax
    sigma = 6.0  # rkspacing / (2.0 * m.sqrt(2.0 * m.log(2.0)))
    k = 33
    rk = k * rkspacing
    l = 8
    epsabs = 0.0
    epsrel = 1.0e-7
    wkspsize = 100000

    bf = basisfn(k, l, Rbins, Thetabins, sigma, rk,
                 epsabs, epsrel, wkspsize)

    r = numpy.linspace(0, Rbins + 1, Rbins, endpoint=False)
    theta = numpy.linspace(-numpy.pi, numpy.pi, Thetabins, endpoint=False)

    r_aug = numpy.append(r, r[-1] + r[1] - r[0])
    theta_aug = numpy.append(theta, theta[-1] + theta[1] - theta[0])

    # Plot using polar projection
    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[10, 1],
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
