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

import vmiutils.pbasex as pbasex
import logging

logger = logging.getLogger('vmiutils.pbasex.fit')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

class PbasexFitDetfn1(pbasex.PbasexFit):
    def fit(self, image, matrix, x=None, y=None, centre=None,
            swapxy=False, section='whole', lmax=None,
            oddl=None, method='least_squares',
            max_iterations=500, tolerance=1.0e-4):

        image_cart = _build_CartesianImage(image, x, y, centre, swapxy)

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

        fit_data(image_polar, matrix, oddl=oddl, lmax=lmax, method=method,
                 tolerance=tolerance, max_iterations=max_iterations)

        self.vmi_image = image_cart.image.copy()
