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
import logging

logger = logging.getLogger('vmiutils.threecolumns')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


def threecolumns_read(filename, delimiter=','):
    """ Read an image from a file containing three columns of values (x, y, z)
    where (x, y) are the pixel coordinates, with x varying fastest, and z is
    the pixel count.

    It is not assumed that the number of pixels in the x and y directions are
    the same, but rather we make a best effort of establishing them from the
    file by looking for recurring values in the first column,

    filename is the filename to read.

    delimiter is the character separating columns.
    """
    logger.debug('reading file: {0}'.format(filename))
    try:
        x, y, z = numpy.loadtxt(filename, unpack=True, delimiter=delimiter)
        logger.debug('number of rows in file: {0}'.format(x.shape[0]))
    except IOError:
        logger.error('could not read file: {0}'.format(filename))
        raise

    # Find the x dimension by looking for the first reappaearance of x[0]
    for idx, val in numpy.ndenumerate(x[1:]):
        if val == x[0]:
            xdim = idx[0] + 1
            break
    logger.debug('dimension in x: {0}'.format(xdim))

    # Hence calculate y dimension
    ydim = x.shape[0] / xdim
    logger.debug('dimension in y: {0}'.format(ydim))

    # Construct vectors of x and y values, and matrix of pixel intensities
    x = x[0:xdim]
    y = y[0::xdim]
    z = z.reshape((xdim, ydim))

    return x, y, z
