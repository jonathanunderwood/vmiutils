import polcart

class VMIPolarImage():
    """ Class used to represent a VMI image stored in polar coordinates
    i.e. in regularly spaced bins in (R, theta)."""

    def __init__(self):
        self.image = None
        self.r = None
        self.theta = None

    def from_VMICartesianImage(self, cimage, radial_bins=None, 
                               angular_bins=None, centre=None, 
                               max_radius=None):
        """Calculate a polar represenation of a VMICartesianImage instance.

        cimage is a VMICartesianImage instance.

        radial_bins and angular_bins specify the desired number of bins in the
        polar representation. If these are none, the number of bins in the
        cartesian image is used.
        
        centre is a tuple

        """
        
        if radial_bins == None:
            radial_bins = cimage.image.shape[0]
        if angular_bins == None:
            angular_bins = cimage.image.shape[1]

        # self.r, self.theta, self.image = polcart.cartesian_to_polar(
        #     cimage.image, cimage.x, cimage.y, radial_bins, angular_bins,
        #     centre, max_radius)
        self.r, self.theta, self.image = polcart.cart2pol(
            cimage.image, cimage.x, cimage.y, centre, radial_bins, angular_bins,
            max_radius)
