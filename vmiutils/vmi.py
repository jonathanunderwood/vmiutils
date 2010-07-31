import numpy
import polcart

class VMICartesianImage():
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

        if x == None:
            self.x = numpy.arange(self.image.shape[0])
        else:
            self.x = x.copy()

        if y == None:
            self.y = numpy.arange(self.image.shape[1])
        else:
            self.y = y.copy()

    def set_centre(self, centre):
        """ Specify the coordinates of the centre of the image as a tuple
        (xcentre, ycentre).
        """
        self.centre = centre

    def from_VMIPolarImage(self, pimage):
        """ Initizize from a VMIPolarImage object by interpolation onto a
        cartesian grid.
        """
        self.x, self.y, self.image = pimage.cartesian_rep()

    def polar_rep(self, radial_bins=None, angular_bins=None, max_radius=None):
        """ Returns a tuple (r, theta, pimage) containing the coordinates and
        polar representation of the image.
        """
        if self.image == None:
            raise ValueError ## FIXME

        if self.centre == None:
            raise ValueError ## FIXME

        if radial_bins == None:
            radial_bins = min(self.image.shape[0], self.image.shape[1]) 

        if angular_bins == None:
            angular_bins = min(self.image.shape[0], self.image.shape[1]) 

        return polcart.cart2pol(self.image, self.x, self.y, self.centre, 
                                radial_bins, angular_bins, max_radius)

    def centre_of_gravity(self):
        """ Returns a tuple representing the coordinates corresponding to the
        centre of gravity of the image.
        """
        xval = self.x * self.image 
        yval = self.y[:,numpy.newaxis] * self.image
        return xval.sum() / self.image.sum(), yval.sum() / self.image.sum()

    def centre_of_grid(self):
        """Returns a tuple containing the central coordinates of the cartesian
        grid."""
        xc = 0.5 * (self.x[-1] - self.x[0])
        yc = 0.5 * (self.y[-1] - self.y[0])
        return xc, yc

class VMIPolarImage():
    """ Class used to represent a VMI image stored in polar coordinates
    i.e. in regularly spaced bins in (r, theta)."""

    def __init__(self):
        self.image = None
        self.r = None
        self.theta = None
        self.rfactor = None

    def from_VMICartesianImage(self, cimage, radial_bins=None, 
                               angular_bins=None, max_radius=None):
        """Calculate a polar represenation of a VMICartesianImage instance.

        cimage is a VMICartesianImage instance.

        radial_bins and angular_bins specify the desired number of bins in the
        polar representation. If these are none, the number of bins in the
        cartesian image is used.
        """
        self.r, self.theta, self.image = \
            cimage.polar_rep(radial_bins, angular_bins, max_radius)

        self.rfactor = self.r[1] - self.r[0]

    def cartesian_rep(self, xbins=None, ybins=None):
        """ Returns a tuple (x, y, image) containing the coordinates and
        cartesian represenation of the image. xbins and ybins optionally
        specify the number of bins in each dimension. If not specified, the
        number of bins in each direction will be equal to the number of radial
        bins in the polar image.
        """
        if xbins == None:
            xbins = self.image.shape[0]

        if ybins == None:
            ybins = self.image.shape[0]

        return polcart.pol2cart(self.image, self.r, self.theta, xbins, ybins)
