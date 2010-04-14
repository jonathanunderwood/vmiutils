# TODO: use numpy.linsapce rather than numpy.arange where appropriate

import math
import numpy
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

import scipy.interpolate as spint



def pol2cart(image, r=None, theta=None, xbins=None, ybins=None, 
             order=3):

    if r == None:
        r = numpy.arange(0.5, image.shape[0])

    rbinw = r[-1] / (image.shape[0] - 1) # Assume equally spaced

    tpts = image.shape[1]
    tbinw = 2.0 * numpy.pi / tpts # Assume equally spaced

    if theta == None:
        theta = numpy.arange(0.5 * tbinw - numpy.pi, 
                             tpts * tbinw - numpy.pi, 
                             tbinw)

    # If the number of bins in the cartesian image is not specified, set it to
    # be the same as the number of radial bins in the polar image
    if xbins == None:
        xbins = image.shape[0]

    if ybins == None:
        ybins = image.shape[0]

    xbinw = 2.0 * r[-1] / (xbins - 1)
    ybinw = 2.0 * r[-1] / (ybins - 1)

    xmin = ymin = -r[-1]
    print xmin, xbinw
    def fmap(out_coord):
        ix, iy = out_coord # Pixel array coords
        x = (ix + 0.5) * xbinw + xmin
        y = (iy + 0.5) * ybinw + ymin
        r = numpy.sqrt(x * x + y * y)
        t = numpy.arctan2(x, y)
        ir = r / rbinw - 0.5
        it = (t + numpy.pi) / tbinw - 0.5
        return ir, it


    import scipy.ndimage
    cimage = scipy.ndimage.geometric_transform(
        image, fmap, order = order, output_shape=(xbins, ybins))

    x = numpy.linspace(-r[-1], r[-1], xbins)
    y = numpy.linspace(-r[-1], r[-1], ybins)

    # print x
    # print y
    print cimage

    return x, y, cimage

def cart2pol(image, x=None, y=None, radial_bins=256, 
             angular_bins=256, centre=None, rmax=None, 
             order=3):
    """ Convert an image on a regularly spaced cartesian grid into a regular
    spaced grid in polar coordinates using interpolation.
    
    x and y contain the x and y coordinates corresponding to each
    bin centre. If either of these are none, unit bin widths are assumed.

    radial_bins and angular_bins define the number of bins in the polar
    representation of the image.

    centre is a tuple containing the x and y coordinates of the image centre
    (does not need to be an integer). If this is set to None the midpoint x
    and y coordinates are used.

    rmax defines the maximum radius from the image centre to consider.

    Here we employ the convention that the angle (theta) is that between the
    second axis (y-axis) and the position vector and that it lies in the range
    [-pi,pi].
    """
    
    if x == None: # Note: these are values at bin centre
        x = numpy.arange(0.5, image.shape[0] + 0.5)

    if y == None: # Note: these are values at bin centre
        y = numpy.arange(0.5, image.shape[1] + 0.5)

    # Centre is the value of the centre coordinate, rather than the
    # pixel number
    if centre == None:
        xc = 0.5 * (x[0] + x[-1])
        yc = 0.5 * (y[0] + y[-1])
    else:
        xc = centre[0]
        yc = centre[1]

    print 'cent', xc, yc

    x = x - xc
    y = y - yc

    # Calculate minimum distance from centre to edge of image - this
    # determines the maximum radius in the polar image. Specifically, rmax is
    # defined as the value of r at the centre of the outermost radial pixel.
    xsize = min(abs(x[0]), x[-1])
    ysize = min(abs(y[0]), y[-1])
    max_rad = min(xsize, ysize)

    if rmax == None:
        rmax = max_rad
    elif rmax > max_rad:
        raise ValueError

    # Polar image bin widths
    rbinw = rmax / (radial_bins - 0.5)
    tbinw = (2.0 * math.pi) / angular_bins

    # Cartesian image bin widths
    xbinw = (x[-1] - x[0]) / (image.shape[0] - 1)
    ybinw = (y[-1] - y[0]) / (image.shape[1] - 1)

    def fmap2(out_coord):
        ir, it = out_coord # In pixel units
        # Find values of r and theta at pixel centre - need to add 0.5.
        r = (ir + 0.5) * rbinw
        t = (it + 0.5) * tbinw - numpy.pi
        # Find corresponding x and y coords to pixel centre.
        x = r * numpy.sin(t) + xc
        y = r * numpy.cos(t) + yc
        # Find corresponding x and y fractional array indices.  Subtract 0.5
        # here since the interpolation function assumes the bin value is at
        # the value of the array index, not the bin centre.
        ix = (x / xbinw) - 0.5
        iy = (y / ybinw) - 0.5
#        print ix, iy
        return ix, iy

    import scipy.ndimage
    pimage = scipy.ndimage.geometric_transform(
        image, fmap2, order = order, output_shape=(radial_bins, angular_bins))


    r = numpy.linspace(0.5 * rbinw, rmax, radial_bins)
    t = numpy.linspace(0.5 * tbinw - numpy.pi, 
                       (angular_bins -0.5) * tbinw - numpy.pi, 
                       angular_bins) 
    print r
    print t
    print tbinw, t[1]-t[0]
    return r, t, pimage
    

