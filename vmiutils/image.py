import numpy
from numpy.linalg import lstsq
import polcart
import logging
from scipy.special import lpn as legpol

logger = logging.getLogger('vmiutils.image')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)

def _round_int(x):
    return int(round(x))

class CartesianImage():
    """ Class used to represent a VMI image stored as a cartesian
    array.
    """
    def __init__(self):
        self.image = None
        self.x = None
        self.y = None
        self.xbinw = None
        self.ybinw = None
        self.centre = None
        self.quadrant = None
        
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
        
        # Set bin widths in each dimension assuming bins are equally spaced
        self.xbinw = self.x[1] - self.x[0]
        self.ybinw = self.y[1] - self.y[0]

    def set_centre(self, centre):
        """ Specify the coordinates of the centre of the image as a tuple
        (xcentre, ycentre). These coordinates are not expected to be integers
        (though they can be).
        """
        self.centre = centre
        
        # Define the centre pixel of the image by rounding to the nearest
        # pixel 
        cx = _round_int((centre[0] - self.x[0]) / self.xbinw)
        cy = _round_int((centre[0] - self.y[0]) / self.ybinw)
        self.centre_pixel = (cx, cy)

        # Set up views of quadrants. The quadrants are numbered 0-3:
        # Quadrant 0: from centre to (xmax, ymax) [Top right]
        # Quadrant 1: from centre to (xmax, 0)    [Bottom right]
        # Quadrant 2: from centre to (0, 0)       [Bottom Left]
        # Quadrant 3: from centre to (0, ymax]    [Top left]
        self.quadrant = [
        self.image[cx::, cy::], 
        self.image[cx::, cy - 1::-1],
        self.image[cx - 1::-1, cy - 1::-1],
        self.image[cx - 1::-1, cy::]
        ]
        
    def get_quadrant(self, quad):
        """ Return a numpy array instance containing the requested image
        quadrant indexed such that [0, 0] is the image centre, and increasing
        x and y move away from the centre.
        """
        if quad in ('upper right', 'top right', 'ur', 0):
            return self.quadrant[0]
        elif quad in ('lower right', 'bottom right', 'lr', 1):
            return self.quadrant[1]
        elif quad in ('lower left', 'bottom left', 'll', 2):
            return self.quadrant[2]
        elif quad in ('upper left', 'top left', 'ul', 3):
            return self.quadrant[3]
        else:
            logger.error('quad argument not recognized')
            raise ValueError

    def zoom_circle(self, rmax):
        """ Return a new CartesianImage instance containing a square section
        centred on the image centre and containing the circular section of the
        image specified by rmax in image coordinates (not bins).
        """
        xminb = _round_int((self.centre[0] - rmax) / self.xbinw)
        xmaxb = _round_int((self.centre[0] + rmax) / self.xbinw)

        yminb = _round_int((self.centre[1] - rmax) / self.ybinw)
        ymaxb = _round_int((self.centre[1] + rmax) / self.ybinw)
        
        return self.zoom_rect_pix([xminb, xmaxb, yminb, ymaxb])

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
            z.set_center(self.center)
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

    def radial_spectrum(self):
        """ Return a tuple (r, intensity) for the radial spectrum calculated
        by summing over theta for each r bin.
        """
        return self.r, self.image.sum(1)

    def beta_coefficients(self, lmax=2, oddl=False):
        """ Return a tuple (r, beta) representing the values of the beta
        parameters at for each radial bin calculated by fitting to an
        expansion in Legendre polynomials up to order lmax. oddl specifies
        whether odd l coefficients are fit or not.
        """

        costheta = numpy.cos(self.theta)
        A = numpy.c_[[legpol(lmax, ct)[0] for ct in costheta]]
        logger.debug(
            'matrix calculated for beta fitting with shape {0}'.format(A.shape))

        if oddl is False:
            A = A[:, ::2]
            logger.debug(
                'odd l coefs not fit: matrix reduced to shape {0}'.format(A.shape))

        try:
            # TODO set rcond
            beta, resid, rank, s = lstsq(A, self.image.transpose())
            # Note that beta is indexed as beta[l, r]
            # TODO: do something with resid, rank, s
        except numpy.linalg.LinAlgError:
            logger.error(
                'failed to converge while fitting beta coefficients')
            raise
        logger.debug('beta coefficents fit successfully')


        # Normalize to beta_0 = 1 at each r
        beta0 = beta[0, :]
        beta = beta / beta0
        logger.debug('beta coefficents normalized')

        if oddl is False:
            logger.debug('adding rows to beta matrix for odd l coeffs')
            logger.debug('beta shape before adding new rows {0}'.format(beta.shape)) 
            beta = numpy.insert(beta, numpy.arange(1, lmax), 0, axis=0)
            logger.debug('rows for odd l added to beta array; new shape {0}'.format(beta.shape)) 

        return self.r, beta
