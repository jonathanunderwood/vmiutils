import numpy
import polcart
import logging

logger = logging.getLogger('vmiutils.image')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)



class CartesianImage():
    """ Class used to represent a VMI image stored as a cartesian
    array.
    """
    def __init__(self):
        self.image = None
        self.x = None
        self.y = None
        self.centre = None

    def from_numpy_array(self, image, x=None, y=None):
        """ Initialize from an image stored in a numpy array. If x or y are
        not specified, the x and y coordinates are stored as pixel values.
        """
        self.image = image.copy()

        if x is None:
            self.x = numpy.arange(self.image.shape[0])
        else:
            self.x = x.copy()

        if y is None:
            self.y = numpy.arange(self.image.shape[1])
        else:
            self.y = y.copy()

    def set_centre(self, centre):
        """ Specify the coordinates of the centre of the image as a tuple
        (xcentre, ycentre).
        """
        self.centre = centre

    def zoom_circle(self, rmax):
        """ Return a new CartesianImage instance containing a square section
        centred on the image centre and containing the circular section of the
        image specified by rmax in image coordinates (not bins).
        """
        xbinw = self.x[1] - self.x[0]
        ybinw = self.y[1] - self.y[0]

        xminb = (self.centre[0] - rmax) / xbinw
        xmaxb = (self.centre[0] + rmax) / xbinw

        yminb = (self.centre[1] - rmax) / ybinw
        ymaxb = (self.centre[1] + rmax) / ybinw
        
        return self.zoom_rect_pix([xminb, xmaxb, yminb, ymaxb])

    def zoom_rect_coord(self, rect):
        xbinw = self.x[1] - self.x[0]
        ybinw = self.y[1] - self.y[0]

        xminb = (rect[0] - self.x[0]) / xbinw
        xmaxb = (rect[1] - self.x[0]) / xbinw

        yminb = (rect[2] - self.y[0]) / ybinw
        ymaxb = (rect[3] - self.y[0]) / ybinw
    
        return self.zoom_rect_pix([xminb, xmaxb, yminb, ymaxb])
        
    def zoom_rect_pix(self, rect):
        """ Return a new CartesianImage instance containing the zoomed image
        specified by rect. 

        rect is a list containing the rectanlge to zoom specified in terms of
        bins: [xmin, xmax, ymin, ymax]. As such, all elements of rect should
        be integer.
        """
        try:
            xmin = rect[0]
            xmax = rect[1]
            ymin = rect[2]
            ymax = rect[3]
            z = CartesianImage()
            z.from_numpy_array(self.image[xmin:xmax, ymin:ymax],
                               self.x[xmin:xmax], self.y[ymin:ymax])
            return z
        except IndexError:
            logger.error('rect outside image')
            raise
        except TypeError:
            logger.error('rect must be a list of integers (bins)')
            raise

    def transpose(self):
        self.image = self.image.transpose()

    def from_PolarImage(self, pimage):
        """ Initizize from a PolarImage object by interpolation onto a
        cartesian grid.
        """
        self.x, self.y, self.image = pimage.cartesian_rep()

    def polar_rep(self, rbins=None, thetabins=None, rmax=None):
        """ Returns a tuple (r, theta, pimage) containing the coordinates and
        polar representation of the image.

        rbins and thetabins specify the number of bins in the returned image.

        rmax specifies the maximum radius to consider, and is specified in the
        coordinate system of the image (as opposed to bin number). If rmax is
        None, then the largest radius possible is used.
        """
        if self.image is None:
            logger.error('no image data')
            raise ValueError ## FIXME

        if self.centre is None:
            logger.error('image centre not defined')
            raise ValueError ## FIXME

        if rbins is None:
            rbins = min(self.image.shape[0], self.image.shape[1]) 

        if thetabins is None:
            thetabins = min(self.image.shape[0], self.image.shape[1]) 

        return polcart.cart2pol(self.image, self.x, self.y, self.centre, 
                                rbins, thetabins, rmax)

    def centre_of_gravity(self):
        """Returns a tuple representing the coordinates corresponding to the
        centre of gravity of the image."""
        xval = self.x * self.image 
        yval = self.y[:,numpy.newaxis] * self.image
        return xval.sum() / self.image.sum(), yval.sum() / self.image.sum()

    def centre_of_grid(self):
        """Returns a tuple containing the central coordinates of the cartesian
        grid."""
        xc = 0.5 * (self.x[-1] - self.x[0])
        yc = 0.5 * (self.y[-1] - self.y[0])
        return xc, yc

class PolarImage():
    """ Class used to represent a VMI image stored in polar coordinates
    i.e. in regularly spaced bins in (r, theta)."""

    def __init__(self):
        self.image = None
        self.r = None
        self.theta = None
        self.rbins = None
        self.thetabins = None

    def from_numpy_array(self, image, r=None, theta=None):
        """ Initialize from a polar image stored in a numpy array. If R or theta are
        not specified, the r and theta coordinates are stored as pixel values.
        """
        self.image = image.copy()

        if r is None:
            self.r = numpy.arange(self.image.shape[0])
        else:
            self.r = r.copy()

        if theta is None:
            self.theta = numpy.linspace(-numpy.pi, numpy.pi, self.image.shape[1])
        else:
            self.theta = theta.copy()

    def from_CartesianImage(self, cimage, rbins=None, 
                               thetabins=None, rmax=None):
        """Calculate a polar represenation of a CartesianImage instance.

        cimage is a CartesianImage instance.

        rbins and thetabins specify the desired number of bins in the
        polar representation. If these are none, the number of bins in the
        cartesian image is used.
        """
        self.r, self.theta, self.image = \
            cimage.polar_rep(rbins, thetabins, rmax)

        self.rbins = self.r.shape[0]
        self.thetabins = self.theta.shape[0]

    def cartesian_rep(self, xbins=None, ybins=None):
        """ Returns a tuple (x, y, image) containing the coordinates and
        cartesian represenation of the image. xbins and ybins optionally
        specify the number of bins in each dimension. If not specified, the
        number of bins in each direction will be equal to the number of radial
        bins in the polar image.
        """
        if xbins is None:
            xbins = self.image.shape[0]

        if ybins is None:
            ybins = self.image.shape[0]

        return polcart.pol2cart(self.image, self.r, self.theta, xbins, ybins)
