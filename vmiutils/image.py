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
import numpy.linalg
import numpy.polynomial.legendre as legendre
import polcart
import logging
import scipy.ndimage

import copy
import pickle
import matplotlib
import matplotlib.cm

logger = logging.getLogger('vmiutils.image')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


def _round_int(x):
    return int(round(x))


class CartesianImage():
    _metadata = ['xbinw', 'ybinw', 'centre', 'shape', 'quad']
    _numpydata = ['x', 'y', 'image']

    """Class used to represent a VMI image stored as a cartesian array.

    image specifies the image data. If image is a 2D numpy ndarray,
    this will be stored in the returned instance. If image is any of
    the strings "empty" or "Empty" a 2D numpy.empty ndarray is
    created. If image is either of the strings "zeros" or "Zeros", a
    2D numpy.zeros ndarray is created. If image is not specified, or
    None, the image data is not initialized.

    if image is "empty", "Empty, "zeros" or "Zeros", xbins and ybins
    specify the size of the image to create.

    x specifies the x coordinates of the image data. If no x argument
    is specified, the bin number is used.

    y specifies the y coordinates of the image data. If no y argument
    is specified, the bin number is used.

    centre specifies the centre of the image. If not specified, the
    centre coordinate of the image array is used.
    """

    def __init__(self, image=None, x=None, y=None,
                 xbins=None, ybins=None, centre=None):

        self.from_numpy_array = self.__from_numpy_array
        self.from_no_data = self.__from_no_data
        self.from_PolarImage = self.__from_PolarImage

        if image is None:
            self.image = None
            self.x = None
            self.y = None
            self.xbinw = None
            self.ybinw = None
            self.centre = None
            self.shape = None
            self.quad = None
            return

        elif image in ('empty', 'Empty', 'zeros', 'Zeros'):
            if x is not None and y is not None:
                self.x = x.copy()
                self.y = y.copy()
                xbins = x.shape[0]
                ybins = y.shape[0]
            elif xbins is not None and ybins is not None:
                self.x = numpy.arange(float(xbins))
                self.y = numpy.arange(float(ybins))
            else:
                logger.error(
                    'x and y dimensions of CartesianImage not specified')

            if image in ('empty', 'Empty'):
                self.image = numpy.empty((xbins, ybins))
            else:
                self.image = numpy.zeros((xbins, ybins))

        elif isinstance(image, numpy.ndarray):
            self.image = image.copy()

            if x is None:
                self.x = numpy.arange(float(self.image.shape[0]))
            else:
                self.x = x.copy()

            if y is None:
                self.y = numpy.arange(float(self.image.shape[1]))
            else:
                self.y = y.copy()

        self.shape = self.image.shape

        # Set bin widths in each dimension assuming bins are equally
        # spaced
        self.xbinw = self.x[1] - self.x[0]
        self.ybinw = self.y[1] - self.y[0]

        if centre is None or centre == 'grid_centre':
            self.set_centre(self.centre_of_grid())
        elif centre == 'cofg':
            self.set_centre(self.centre_of_gravity())
        else:
            self.set_centre(centre)

    @classmethod
    def from_numpy_array(cls, array, x=None, y=None,
                         xbins=None, ybins=None, centre=None):
        instance = cls()
        instance.from_numpy_array(array, x=x, y=y, xbins=xbins,
                                  ybins=ybins, centre=centre)
        return instance

    def __from_numpy_array(self, array, x=None, y=None,
                           xbins=None, ybins=None, centre=None):
        self.image = image.copy()

        if x is None:
            self.x = numpy.arange(float(self.image.shape[0]))
        else:
            self.x = x.copy()

        if y is None:
            self.y = numpy.arange(float(self.image.shape[1]))
        else:
            self.y = y.copy()

        self.shape = self.image.shape

        # Set bin widths in each dimension assuming bins are equally
        # spaced
        self.xbinw = self.x[1] - self.x[0]
        self.ybinw = self.y[1] - self.y[0]

        if centre is None or centre == 'grid_centre':
            self.set_centre(self.centre_of_grid())
        elif centre == 'cofg':
            self.set_centre(self.centre_of_gravity())
        else:
            self.set_centre(centre)

    def __from_no_data(self, image=None, x=None, y=None,
                       xbins=None, ybins=None, centre=None):
        if x is not None and y is not None:
            self.x = x.copy()
            self.y = y.copy()
            xbins = x.shape[0]
            ybins = y.shape[0]
        elif xbins is not None and ybins is not None:
            self.x = numpy.arange(float(xbins))
            self.y = numpy.arange(float(ybins))
        else:
            logger.error(
                'x and y dimensions of CartesianImage not specified')

        if image in ('empty', 'Empty'):
            self.image = numpy.empty((xbins, ybins))
        else:
            self.image = numpy.zeros((xbins, ybins))

        self.shape = self.image.shape

        # Set bin widths in each dimension assuming bins are equally
        # spaced
        self.xbinw = self.x[1] - self.x[0]
        self.ybinw = self.y[1] - self.y[0]

        if centre is None or centre == 'grid_centre':
            self.set_centre(self.centre_of_grid())
        elif centre == 'cofg':
            self.set_centre(self.centre_of_gravity())
        else:
            self.set_centre(centre)

    def copy(self):
        return copy.copy(self)

    def set_centre(self, centre):
        """Specify the coordinates of the centre of the image as a tuple
        (xcentre, ycentre). These coordinates are not expected to be
        integers (though they can be). These coordinates are in the
        same units as the image x and y array, not image array bin
        numbers.

        This function also sets up the quadrant views for the
        image. These quadrants are taken about the lower left corner
        of the bin containing the image centre. As such, they are
        approximate (i.e. out by some fracton of a bin). If exact
        quadrants about the centre are required, it is necessary to
        call the resample method of the class.

        """
        self.centre = centre

        # Define the centre pixel of the image by rounding to the nearest
        # pixel.
        cx = _round_int((centre[0] - self.x[0]) / self.xbinw)
        cy = _round_int((centre[1] - self.y[0]) / self.ybinw)
        self.centre_pixel = (cx, cy)

        # Set up slices to give views of quadrants. The quadrants are numbered
        # 0-3:
        # Quadrant 0: from centre to (xmax, ymax) [Top right]
        # Quadrant 1: from centre to (xmax, 0) [Bottom right]
        # Quadrant 2: from centre to (0, 0) [Bottom Left]
        # Quadrant 3: from centre to (0, ymax] [Top left]
        #
        # Using the centre pixel values to demark the quadrants
        # approximates the centre as lying in the bottom left corner
        # of the pixel containing the image centre.
        # self.quadrant = [
        # self.image[cx::, cy::],
        # self.image[cx::, cy - 1::-1],
        # self.image[cx - 1::-1, cy - 1::-1],
        # self.image[cx - 1::-1, cy::]
        # ]

        self.quad = [
            (slice(cx, None, None), slice(cy, None, None)),
            (slice(cx, None, None), slice(cy - 1, None, -1)),
            (slice(cx - 1, None, -1), slice(cy - 1, None, -1)),
            (slice(cx - 1, None, -1), slice(cy, None, None))
        ]

    def resample(self, xbins=None, ybins=None, align_centre=False, order=3):
        """Resample the image onto a new grid.

        If xbins and/or ybins is not None (the default) they specify
        the dimensions required. If these are None, the exisiting
        number of bins in that dimension is used.

        If align_centre is True, the new grid will be shifted so that
        current centre lies on a lower left corner of the nearest
        bin. This ensures the quadrants are exact. If align_centre is
        False (the default) the quadrants are taken about the nearest
        pixel to the image centre.

        order specifies the interpolation order used when resampling.
        """
        x0 = self.centre[0]
        y0 = self.centre[1]

        # If we are to resample the image to (xbins x ybins), set
        # up the new x and y details now. For the moment this
        # doesn't include any offset to put the centre on a pixel
        # boundary - that is taken in to account subsequently
        old_xmin = self.x[0]
        old_xmax = self.x[-1] + self.xbinw
        old_xbinw = self.xbinw

        if xbins is not None:
            self.xbins = xbins
            self.xbinw = (old_xmax - old_xmin) / xbins
            self.x = numpy.linspace(old_xmin, old_xmax, xbins,
                                    endpoint=False)

        old_ymin = self.y[0]
        old_ymax = self.y[-1] + self.ybinw
        old_ybinw = self.ybinw

        if ybins is not None:
            self.ybins = ybins
            self.ybinw = (old_ymax - old_ymin) / ybins
            self.y = numpy.linspace(old_ymin, old_ymax, ybins,
                                    endpoint=False)

        # Find the pixel in the new grid whose lower left corner
        # is closest to the centre
        cx = _round_int((x0 - self.x[0]) / self.xbinw)
        cy = _round_int((y0 - self.y[0]) / self.ybinw)

        if align_centre is True:
            # We want to shift the grid such that the image centre lies at
            # the bottom left corner of a pixel, such that when we form
            # the quadrants later, they are exact.
            xoffset = x0 - self.x[cx]
            yoffset = y0 - self.y[cy]

            self.x += xoffset
            self.y += yoffset
        else:
            xoffset = 0.0
            yoffset = 0.0

        # Now we need a function which maps the new image bin
        # coords to the old image bin coordinates (for
        # geometric_transform to use to interpolate the new
        # grid). For eficiency we define a couple of new
        # variables.
        xbinw_ratio = self.xbinw / old_xbinw
        ybinw_ratio = self.ybinw / old_ybinw
        xoffset_bins = xoffset / self.xbinw
        yoffset_bins = yoffset / self.ybinw

        def _map(coords):
            # Below, for reference is the algorithm without
            # optimization. It should be pretty obvious that this
            # reduces to the simple formulas that follow
            #
            # Calculate the x and y values corresponding to this
            # pxiel in the new grid
            # x = coords[0] * self.xbinw + self.x[0]
            # y = coords[1] * self.ybinw + self.y[0]
            # Calculate the fractional pixel location of this x
            # and y in the old grid
            # old_x = (x - old_xmin) / old_xbinw
            # old_y = (y - old_ymin) / old_ybinw
            # return (old_x, old_y)
            #
            old_x = (coords[0] + xoffset_bins) * xbinw_ratio
            old_y = (coords[1] + yoffset_bins) * ybinw_ratio
            return (old_x, old_y)

        # Here we use mode='nearest' because the shift of the grid
        # by at most half a pixel will move one side outside of
        # the original grid by at most half a pixel. This is an
        # acceptable compromise.
        self.image = scipy.ndimage.geometric_transform(
            self.image, _map, order=order,
            output_shape=(self.x.shape[0], self.y.shape[0]),
            mode='nearest',
        )

        self.centre_pixel = (cx, cy)

        # Set up slices to give views of quadrants. The quadrants are numbered
        # 0-3:
        # Quadrant 0: from centre to (xmax, ymax) [Top right]
        # Quadrant 1: from centre to (xmax, 0) [Bottom right]
        # Quadrant 2: from centre to (0, 0) [Bottom Left]
        # Quadrant 3: from centre to (0, ymax] [Top left]

        self.quad = [
            (slice(cx, None, None), slice(cy, None, None)),
            (slice(cx, None, None), slice(cy - 1, None, -1)),
            (slice(cx - 1, None, -1), slice(cy - 1, None, -1)),
            (slice(cx - 1, None, -1), slice(cy, None, None))
        ]

    def __quad_idx(self, quad):
        if quad in ('upper right', 'top right', 'ur', 0):
            return 0
        elif quad in ('lower right', 'bottom right', 'lr', 1):
            return 1
        elif quad in ('lower left', 'bottom left', 'll', 2):
            return 2
        elif quad in ('upper left', 'top left', 'ul', 3):
            return 3
        else:
            logger.error('quad argument not recognized: {0}'.format(quad))
            raise ValueError

    def get_quadrant(self, quad):
        """ Return a numpy array instance containing the requested image
        quadrant indexed such that [0, 0] is the image centre, and increasing
        |x| and |y| as indices increase.
        """
        return self.image[self.quad[self.__quad_idx(quad)]]

    def set_quadrant(self, quad, data):
        """ Return a numpy array instance containing the requested image
        quadrant indexed such that [0, 0] is the image centre, and increasing
        x and y move away from the centre.
        """
        qslice = self.quad[self.__quad_idx(quad)]

        if self.image[qslice].shape != data.shape:
            logger.error('data not correct shape for specified quadrant')
            raise ValueError
        else:
            self.image[qslice] = data

    def zoom_circle(self, rmax, pad=False):
        """ Return a new CartesianImage instance containing a square section
        centred on the image centre and containing the circular section of the
        image specified by rmax in image coordinates (not bins).
        """
        if self.centre is None:
            logger.error(
                'image centre has not been defined prior to asking for zoom_circle')
            raise RuntimeError('image centre undefined')

        xminb = _round_int((self.centre[0] - rmax - self.x[0]) / self.xbinw)
        if xminb < 0 and pad is False:
            logger.error('xminb less than zero in zoom_circle')
            raise RuntimeError('xminb less than zero')

        xmaxb = _round_int((self.centre[0] + rmax - self.x[0]) / self.xbinw)
        if xmaxb > self.image.shape[0] and pad is False:
            logger.error('xmaxb greater than image size in zoom_circle')
            raise RuntimeError('xmaxb greater than image size')

        yminb = _round_int((self.centre[1] - rmax - self.y[0]) / self.ybinw)
        if yminb < 0 and pad is False:
            logger.error('yminb less than zero in zoom_circle')
            raise RuntimeError('yminb less than zero')

        ymaxb = _round_int((self.centre[1] + rmax - self.y[0]) / self.ybinw)
        if ymaxb > self.image.shape[0] and pad is False:
            logger.error('ymaxb greater than image size in zoom_circle')
            raise RuntimeError('ymaxb greater than image size')

        return self.zoom_rect_pix([xminb, xmaxb, yminb, ymaxb], pad=pad)

    def zoom_rect_coord(self, rect):
        """ Return a new CartesianImage instance containing the zoomed image
        specified by rect. 

        rect is a list containing the rectanlge to zoom specified in
        coordinates: [xmin, xmax, ymin, ymax].
        """
        xminb = _round_int((rect[0] - self.x[0]) / self.xbinw)
        xmaxb = _round_int((rect[1] - self.x[0]) / self.xbinw)

        yminb = _round_int((rect[2] - self.y[0]) / self.ybinw)
        ymaxb = _round_int((rect[3] - self.y[0]) / self.ybinw)

        return self.zoom_rect_pix([xminb, xmaxb, yminb, ymaxb])

    def zoom_rect_pix(self, rect, pad=False):
        """Return a new CartesianImage instance containing the zoomed image
        specified by rect. 

        rect is a list containing the rectanlge to zoom specified in terms of
        bins: [xmin, xmax, ymin, ymax]. As such, all elements of rect should
        be integer.

        if pad is True, then if any of the requested area doesn't lie
        within the image data, then where no data is available, 0s are
        substituted.
        """
        xmin = rect[0]
        xmax = rect[1]
        ymin = rect[2]
        ymax = rect[3]

        if xmin >= 0:
            x1 = xmin
            xstart = 0
        else:
            if pad == True:
                x1 = 0
                xstart = -xmin
            else:
                logger.error('xmin outside of image in zoom_rect_pix')
                raise RuntimeError('xmin outside of image')

        if xmax <= self.image.shape[0]:
            x2 = xmax
        else:
            if pad == True:
                x2 = self.image.shape[0] - 1
            else:
                logger.error('xmax outside of image in zoom_rect_pix')
                raise RuntimeError('xmax outside of image')

        if ymin >= 0:
            y1 = ymin
            ystart = 0
        else:
            if pad == True:
                y1 = 0
                ystart = -ymin
            else:
                logger.error('ymin outside of image in zoom_rect_pix')
                raise RuntimeError('ymin outside of image')

        if ymax <= self.image.shape[0]:
            y2 = ymax
        else:
            if pad == True:
                y2 = self.image.shape[0] - 1
            else:
                logger.error('ymax outside of image in zoom_rect_pix')
                raise RuntimeError('ymax outside of image')

        newimg = numpy.zeros((xmax - xmin + 1, ymax - ymin + 1))

        newimg[xstart:xstart + (x2 - x1),
               ystart:ystart + (y2 - y1)] = self.image[x1:x2, y1:y2]

        newx = numpy.linspace(
            xmin * self.xbinw, xmax * self.xbinw, newimg.shape[0])
        newy = numpy.linspace(
            ymin * self.ybinw, ymax * self.ybinw, newimg.shape[1])

        return CartesianImage(image=newimg, x=newx, y=newy, centre=self.centre)

    @classmethod
    def from_PolarImage(cls, pimage, xbins=None, ybins=None, order=3):
        instance = cls()
        instance.from_PolarImage(pimage, xbins, ybins, order)
        return instance

    def __from_PolarImage(self, pimage, xbins=None, ybins=None, order=3):
        """Initialise from a PolarImage object by interpolation onto a
        cartesian grid.

        xbins and ybins specify the desired number of x and y bins. If
        these are None, the number of bins in each direction will
        be equal to the number of radial bins in the polar image.

        order specifies the interpolaton order used in the conversion.

        """
        self.x, self.y, self.image = pimage.cartesian_rep(xbins, ybins, order)
        self.shape = self.image.shape
        self.xbinw = self.x[1] - self.x[0]
        self.ybinw = self.y[1] - self.y[0]
        self.set_centre(self.centre_of_grid())

    def polar_rep(self, rbins=None, thetabins=None, rmax=None, order=3):
        """ Returns a tuple (r, theta, pimage) containing the coordinates and
        polar representation of the image.

        rbins and thetabins specify the number of bins in the returned image.

        rmax specifies the maximum radius to consider, and is specified in the
        coordinate system of the image (as opposed to bin number). If rmax is
        None, then the largest radius possible is used.

        order specifies the interpolaton order used in the conversion.
        """
        if self.image is None:
            logger.error('no image data')
            raise ValueError  # FIXME

        if self.centre is None:
            logger.error('image centre not defined')
            raise ValueError  # FIXME

        if rbins is None:
            rbins = min(self.image.shape[0], self.image.shape[1])

        if thetabins is None:
            thetabins = min(self.image.shape[0], self.image.shape[1])

        return polcart.cart2pol(self.image, x=self.x, y=self.y,
                                centre=self.centre, radial_bins=rbins,
                                angular_bins=thetabins, rmax=rmax,
                                order=order)

    def centre_of_gravity(self):
        """Returns a tuple representing the coordinates corresponding to the
        centre of gravity of the image."""
        xval = self.x * self.image
        yval = self.y[:, numpy.newaxis] * self.image
        return xval.sum() / self.image.sum(), yval.sum() / self.image.sum()

    def centre_of_grid(self):
        """Returns a tuple containing the central coordinates of the cartesian
        grid."""
        xbinw = self.x[1] - self.x[0]
        ybinw = self.y[1] - self.y[0]
        xc = 0.5 * (self.x[-1] + xbinw - self.x[0])
        yc = 0.5 * (self.y[-1] + ybinw - self.y[0])
        return xc, yc

    def dump(self, fd):
        """ Dump instance of this class to a file descriptor fd.
        """
        for object in self._metadata:
            pickle.dump(getattr(self, object), fd, protocol=2)

        for object in self._numpydata:
            numpy.save(fd, getattr(self, object))

    def load(self, fd):
        """ Load a previously dumped instance of this class from a file
        descriptor fd. """
        for object in self._metadata:
            setattr(self, object, pickle.load(fd))

        for object in self._numpydata:
            setattr(self, object, numpy.load(fd))

    def _augment(self, arr):
        return numpy.append(arr, arr[-1] + arr[1] - arr[0])

    def plot(self, axis, cmap=matplotlib.cm.spectral,
             rasterized=True, transpose=False, clip=None,
             origin_at_centre=False):

        if transpose is False:
            x = self.x
            y = self.y
            image = self.image.T
            xc = self.centre[0]
            yc = self.centre[1]
        elif transpose is True:
            x = self.y
            y = self.x
            image = self.image
            xc = self.centre[1]
            yc = self.centre[0]
        else:
            raise ValueError('transpose must be True or False')

        if origin_at_centre is True:
            x = x - xc
            y = y - yc

        if clip is not None:
            image = image.clip(clip)

        return axis.pcolormesh(self._augment(x),
                               self._augment(y),
                               image, cmap=cmap,
                               rasterized=rasterized)


class PolarImage():

    """ Class used to represent a VMI image stored in polar coordinates
    i.e. in regularly spaced bins in (r, theta)."""

    def __init__(self):
        self.from_numpy_array = self.__from_numpy_array
        self.from_CartesianImage = self.__from_CartesianImage

        self.image = None
        self.r = None
        self.theta = None
        self.rbins = None
        self.thetabins = None

    @classmethod
    def from_numpy_array(cls, image, r=None, theta=None):
        instance = cls()
        instance.from_numpy_array(image, r=r, theta=theta)
        return instance

    def __from_numpy_array(self, image, r=None, theta=None):
        """ Initialize from a polar image stored in a numpy array. If R or theta are
        not specified, the r and theta coordinates are stored as pixel values.
        """
        self.image = image.copy()

        if r is None:
            self.r = numpy.arange(float(self.image.shape[0]))
        else:
            self.r = r.copy()

        if theta is None:
            self.theta = numpy.linspace(
                -numpy.pi, numpy.pi, self.image.shape[1])
        else:
            self.theta = theta.copy()

    @classmethod
    def from_CartesianImage(cls, cimage, rbins=None,
                            thetabins=None, rmax=None, order=3):
        instance = cls()
        instance.from_CartesianImage(cimage, rbins=rbins,
                                     thetabins=thetabins,
                                     rmax=rmax, order=order)
        return instance

    def __from_CartesianImage(self, cimage, rbins=None,
                              thetabins=None, rmax=None, order=3):
        """Calculate a polar represenation of a CartesianImage instance.

        cimage is a CartesianImage instance.

        rbins and thetabins specify the desired number of bins in the
        polar representation. If these are none, the number of bins in the
        cartesian image is used.
        """
        self.r, self.theta, self.image = \
            cimage.polar_rep(rbins, thetabins, rmax, order)

        self.rbins = self.r.shape[0]
        self.thetabins = self.theta.shape[0]

    def cartesian_rep(self, xbins=None, ybins=None, order=3):
        """ Returns a tuple (x, y, image) containing the coordinates and
        cartesian represenation of the image. 

        xbins and ybins optionally specify the number of bins in each
        dimension. If not specified, the number of bins in each direction will
        be equal to the number of radial bins in the polar image.

        order specifies the interpolaton order used in the conversion.
        """
        if xbins is None:
            xbins = self.image.shape[0]

        if ybins is None:
            ybins = self.image.shape[0]

        return polcart.pol2cart(self.image, self.r, self.theta,
                                xbins, ybins, order)

    def radial_spectrum(self):
        """ Return a tuple (r, intensity) for the radial spectrum calculated
        by summing over theta for each r bin.
        """
        spec = self.image.sum(1)
        spec /= spec.max()
        return self.r, spec

    def beta_coefficients(self, lmax=2, mask_val=1.0e-8):
        '''Return a tuple (r, beta) representing the values of the beta
        parameters at for each radial bin calculated by fitting to an
        expansion in Legendre polynomials up to order lmax.

        The returned beta values are normalized to beta[0] = 1 at each
        radius. So, if beta[0] is small, these values can be very
        large.

        mask_val is used to determine which radii beta values are
        calculated for. If this is None, beta values are returned for
        all radiiu, even where there is no intensity and so they have
        no meaningful physical significance. If mask_val is not none,
        then beta values are calculated only when the radial spectrum
        intensity (when normalized to maximum 1) takes a value greater
        than mask_val, otherwise the beta values are set to invalid
        data. In either case beta is a numpy masked array.

        At present even and odd l Legendre polynomials are
        included. In the future we'll extend this method to allow for
        only even l Legendre polynomials to be included in the fit.

        '''

        # Need to fit to central values of the bins in theta, so
        # construct an array of the central values.
        theta_c = self.theta + 0.5 * (self.theta[1] - self.theta[0])

        beta = legendre.legfit(numpy.cos(theta_c), self.image.T, lmax)

        logger.debug('beta coefficents fit successfully')

        # Normalize to beta_0 = 1 at each r
        beta0 = beta[0, :]
        if mask_val != None:
            # Only calculate beta values normalized to beta_0=1 if the
            # radial spectrum, when normalized to unity maximum, has a
            # value greater than mask_val. Note that beta0.max
            # corresponds to the maximum value of the radial spectrum.
            mask_val = mask_val * beta0.max()
            beta0_masked = numpy.ma.masked_inside(beta0, -mask_val, mask_val)
        else:
            beta0_masked =  numpy.ma.array(beta0, mask=numpy.zeros(beta0.size))

        beta = beta / beta0_masked

        logger.debug('beta coefficents normalized')

        return self.r, beta

if __name__ == "__main__":
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    a = numpy.zeros(64).reshape((8, 8))
    a[4][4] = 1.0

    im1 = CartesianImage(image=a, centre=(4, 4))

    im2 = CartesianImage(image=a, centre=(4, 4))

    im2.set_centre((3.49, 3.49))
    im2.resample(xbins=32, ybins=32)

    print im1.x
    print im2.x

    plt.figure(1)

    plt.subplot(221)
    plt.imshow(im1.image.T, cmap=cm.gist_heat, origin='lower',
               interpolation='none',
               extent=(im1.x[0], im1.x[-1] + im1.xbinw,
                       im1.y[0], im1.y[-1] + im1.ybinw),
               )

    plt.subplot(222)
    plt.imshow(im1.image.T, cmap=cm.spectral, origin='lower',
               interpolation='none',
               extent=(im1.x[0], im1.x[-1] + im1.xbinw,
                       im1.y[0], im1.y[-1] + im1.ybinw),
               )

    plt.subplot(223)
    plt.imshow(im2.image.T, cmap=cm.gist_heat, origin='lower',
               interpolation='none',
               extent=(im2.x[0], im2.x[-1] + im2.xbinw,
                       im2.y[0], im2.y[-1] + im2.ybinw),
               )

    plt.subplot(224)
    plt.imshow(im2.image.T, cmap=cm.spectral, origin='lower',
               interpolation='none',
               extent=(im2.x[0], im2.x[-1] + im2.xbinw,
                       im2.y[0], im2.y[-1] + im2.ybinw),
               )

    plt.show()
