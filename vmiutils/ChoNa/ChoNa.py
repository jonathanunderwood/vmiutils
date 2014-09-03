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


# Functions relating to "Application of Abel inversion in real-time
# calculations for circularly and elliptically symmetric radiation sources",
# Y. T. Cho and S-J Na, Sci. Technol. 16 878-884 (2005).  Note that the i, j
# indices in the Cho and Na paper start from 1 rather than 0. Here we index
# from zero, so the formulas are adjusted accordingly.

import numpy
import logging
import vmiutils.image as vmi

# Set up logging and create a null handler in case the application doesn't
# provide a log handler
logger = logging.getLogger('vmiutils.ChoNa')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

def __P(i, j):
    """
    Calculates the Cho and Na P matrix elements assuming d = 1.
    """
    # The following results in "RuntimeWarning: invalid value encountered in arccos"
    # theta = numpy.where(j >= i, numpy.arccos(i / j + 1.0), 0.0)
    # because the arrcos is evaluated for i and j before the where.
    # See:
    # http://mail.scipy.org/pipermail/numpy-discussion/2013-January/064987.html
    # http://mail.scipy.org/pipermail/numpy-discussion/2013-January/064988.html
    theta = numpy.arccos(numpy.where(j >= i, i / (j + 1.0), 0.0))
    return numpy.where(j >= i, 0.5 * (numpy.power(j + 1.0, 2) * theta -
                                      numpy.power(i, 2) * numpy.tan(theta)),
                       0.0)

def area_matrix(dim):
    """
    Calculate and return a numpy array containing the area matrix as defined
    by Cho and Na. The returned square array is of size dim. The array is
    normalized to unit width.
    """
    P = numpy.fromfunction(__P, (dim, dim))

    # Here we set d = 1 / dim^2 to normalize to a width of 1.  This isn't
    # really necessary. Likewise we could drop the 0.5 in __P above.
    P = numpy.multiply(P, 1.0 / (dim * dim))

    S = P.copy()
    S[0:dim - 1, :] -= P[1:dim, :]
    S[:, 1:dim] -= P[:, 0:dim - 1]
    S[0:dim - 1, 1:dim] += P[1:dim, 0:dim - 1]

    return S

def invert_CartesianImage(image, S=None):
    
    # Find largest size of matrix we need
    dim = max(image.get_quadrant(i).shape[0] for i in xrange(4))

    S = area_matrix(dim)
    logger.debug('S matrix calculated with dim={0}'.format(dim))
        
    image_out = vmi.CartesianImage(
        image='empty', x=image.x, y=image.y, centre=image.centre)

    for i in xrange(4):
        quadrant = image.get_quadrant(i)
        dim = quadrant.shape[0]
        qinv = numpy.linalg.solve(S[0:dim, 0:dim], quadrant)
        image_out.set_quadrant(i, qinv)
        logger.debug('quadrant {0} inverted'.format(i))

    return image_out

import copy

def invert_CartesianImage_plreg(image, iterations=10, initial_guess=None, 
                                reduced_tau=None):
    # reduced_tau lies in range 0..2
    if reduced_tau is None:
        reduced_tau = 1.0
    elif reduced_tau <= 0.0 or reduced_tau >= 2.0:
        logger.error('reduced_tau not in range 0.0 - 2.0')
        raise ValueError

    # Find largest size of matrix we need
    dim = max(image.get_quadrant(i).shape[0] for i in xrange(4))

    Smtx = area_matrix(dim)
    logger.debug('S matrix calculated with dim={0}'.format(dim))
        
    if initial_guess is None or initial_guess in ('zeros', 'Zeros'):
        image_out = vmi.CartesianImage(image='zeros', x=image.x, y=image.y, 
                                       centre=image.centre)
    elif isinstance(initial_guess, vmi.CartesianImage):
        image_out = initial_guess.copy()

    for i in xrange(4):
        quad = image.get_quadrant(i)
        quad_out = image_out.get_quadrant(i)

        dim = quad.shape[0]

        S = Smtx[0:dim, 0:dim]

        St = S.transpose()
        StS = numpy.dot(St, S)

        Snorm = numpy.linalg.norm(S)
        Snorm2 = Snorm * Snorm
        
        tau = reduced_tau / Snorm2

        for y in xrange(image.shape[1]):
            print y
            a = quad_out[:, y]
            b = numpy.dot(St, quad[:, y])
            
            for iter in xrange(iterations):
                a += tau * (b - numpy.dot(StS, a))
                a = a.clip(min=0.0)
            
            quad_out[:, y] = a
            logger.debug('quadrant {0} row {1} inverted'.format(i, y))

        image_out.set_quadrant(i, quad_out)
        logger.debug('quadrant {0} inverted'.format(i))

    return image_out

def invert_plreg(image, iterations, initial_guess=None, tau=None, S=None):

    im = image#.transpose()

    if S == None:
        S = area_matrix(im.shape[1]).transpose()
    elif S.shape[1] != im.shape[1]:
        raise ValueError

    Snorm = numpy.linalg.norm(S)
    Snorm2 = Snorm * Snorm

    if tau == None:
	tau = 1.0 / Snorm2
    elif tau <= 0 or tau >= 2 / Snorm2:
	raise ValueError

    print "tau:", tau

    St = S.transpose()
    StS = numpy.dot(St, S)

    if initial_guess == None:
        a = numpy.ones(shape=im.shape, dtype=float)
    elif initial_guess.shape != im.shape:
	raise ValueError
    else:
	a = initial_guess

    for i in xrange(im.shape[0]):
        print i
        b = numpy.dot(St, im[i])

        # Scale inital guess so that each row gives the same number of hits in
        # the corresponding synthesized image as the in the original image
        if initial_guess == None:
            norm = (numpy.dot(S, a[i])).sum()
            norm = im[i].sum() / norm
            print norm
            a[i] *= norm

        for j in xrange(iterations):
            a[i] += tau * (b - numpy.dot(StS, a[i]))
            a[i] = a[i].clip(min=0.0)

    return a

from scipy import signal

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = numpy.mgrid[-size:size+1, -sizey:sizey+1]
    g = numpy.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)




if __name__ == "__main__":
    import image
    import sys
    import pylab
    import matplotlib.cm
    
    file=sys.argv[1]

    try:
        img = numpy.loadtxt(file)
    except IOError:
        print "Could not read file", file
        sys.exit(74)
        
#    img = img.transpose()

    cofg = image.centre_of_gravity(img)

    wksp = image.workspace_init(img, cofg, 0)

    dist_simp = invert(wksp)
#    dist = invert_plreg(wksp, 1000, initial_guess=dist_simp)
#    dist = invert_plreg(wksp, 1000)

#    dist_simp=blur_image(dist_simp, 3)

    fig=pylab.figure(1)
#    pylab.subplot(1, 3, 1)
#    plt = pylab.imshow(dist[30:, 30:], cmap=matplotlib.cm.gray, origin='lower')
#    plt = pylab.imshow(dist, cmap=matplotlib.cm.gray, origin='lower')
#    fig.colorbar(plt)
    pylab.subplot(1, 2, 1)
    plt2 = pylab.imshow(dist_simp, cmap=matplotlib.cm.gray, origin='lower')
    pylab.subplot(1, 2, 2)
    plt2 = pylab.imshow(wksp, cmap=matplotlib.cm.gray, origin='lower')

    pylab.show()

