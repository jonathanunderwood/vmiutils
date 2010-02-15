import numpy
import polcart

class VMICartesianImage():
    """ Class used to represent a VMI image stored as a cartesian
    array. Instantiation requires passing the image data stored in a numpy
    array.
    """
    def __init__(self):
        self.image = None
        self.x = None
        self.y = None
    
    def from_numpy_array(self, image, x=None, y=None):
        """ Initialize from an image stored in a numpy array. If x or y are
        not specified, the x and y coordinates are stored in pixel values.
        """
        self.image = image.copy()

        if x == None:
            self.x = numpy.arange(self.image.shape[0])
            print self.x.shape
        else:
            self.x = x.copy()

        if y == None:
            self.y = numpy.arange(self.image.shape[1])
        else:
            self.y = y.copy()

    def from_VMIPolarImage(self, pimage):
        """ Initizize from a VMIPolarImage object by interpolation onto a
        cartesian grid.
        """
        self.x, self.y, self.image = polcart.polar_to_cartesian(
            pimage.image, pimage.r, pimage.theta)

    def centre_of_gravity(self):
        """ Returns thea tuple representing the coordinates corresponding to
        the centre of gravity of the image. This is not rounded to the nearest
        integers.
        """
        # Here we make use of the numpy broadcasting rules. Had we not already
        # stored the x and y pixel central values we would have done something
        # like this instead:
        # x, y = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]
        # x += 0.5
        # y += 0.5
        # xval = x * self.image 
        # yval = y * self.image
        xval = self.x.transpose() * self.image 
        yval = self.y * self.image
        imsum = self.image.sum()
        return xval.sum() / imsum, yval.sum() / imsum
