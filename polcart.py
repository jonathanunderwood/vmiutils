# TODO: use numpy.linsapce rather than numpy.arange where appropriate

import math
import numpy
import scipy.interpolate as spint

def cartesian_to_polar(cimage, x=None, y=None, radial_bins=256, 
                       angular_bins=256, centre=None, rmax=None): 
    """ Convert an image on a regularly spaced cartesian grid into a regular
    spaced grid in polar coordinates using interpolation.
    
    x and y contain the x and y coordinates corresponding to each
    bin centre. If either of these are none, bin widths are used.

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
    if x == None or y == None:
        xdim = cimage.shape[0]
        ydim = cimage.shape[1]

        # Note: these are values at bin centre
        x = numpy.arange(0.5 - xc, xdim)
        y = numpy.arange(0.5 - yc, ydim)

    xmin = x[0]
    xmax = x[-1]
    ymin = y[0]
    ymax = y[-1]
    
    xbinw = x[1] - x[0] # bin width - assume equally spaced bins 
    ybinw = y[1] - y[0] # bin width - assume equally spaced bins 

    # centre is the value of the centre coordinate, rather than the pixel
    # number 
    if centre == None:
        xc = 0.5 * (xmax - xmin + xbinw)
        yc = 0.5 * (ymax - ymin + ybinw)
    else:
        xc = centre[0]
        yc = centre[1]

    # Calculate minimum distance from centre to edge of image - this
    # determines the maximum radius in the polar image 
    xsize = min(xmax + (0.5 * xbinw) - xc, abs(xmin - (0.5 * xbinw) - xc))
    ysize = min(ymax + (0.5 * ybinw) - yc, abs(ymin - (0.5 * ybinw) - yc))
    max_rad = min(xsize, ysize)

    if rmax == None:
        rmax = max_rad
    elif rmax > max_rad:
        raise ValueError

    # Set up interpolation - cubic spline with no smoothing by default 
    interp = spint.RectBivariateSpline(x - xc, y - yc, cimage)

    # Polar image bin widths
    rbinw = rmax / radial_bins
    tbinw = (2.0 * math.pi) / angular_bins

    # Calculate x and y positions corresponding to the polar bins
    r, theta = numpy.ogrid[0.5 * rbinw:radial_bins * rbinw:rbinw, 
                           0.5 * tbinw - math.pi:angular_bins * tbinw:tbinw]

    _x = r * numpy.sin(theta)
    _y = r * numpy.cos(theta)

    # Calculate image in polar representation
    pimage = interp.ev(_x.ravel(), _y.ravel()).reshape(_x.shape)
    return r.flatten(), theta.flatten(), pimage

def polar_to_cartesian(pimage, r=None, theta=None, xbins=None, zbins=None):
    """ Convert an image stored on a grid regularly spaced in polar
    coordinates to cartesian coordinates.

    pimage contains the image data.

    r and theta optionally contain the central radial and angular coordinates
    for each bin in pimage. If None we use bin coordinates.

    xbins and zbins optionally specify the number of bins in each cartesian
    coordinate for the returned cartesian image. If None, these are set to be
    equal to the number of radial bins in the input polar image.
    """
    if r == None:
        r = numpy.arange(0.5, pimage.shape[0])
    #FIXME: check if r has correct dimension if passed in

    if theta == None:
        tbinw = 2.0 * math.pi / pimage.shape[1]
        #FIXME: CHECK THE LINE BELOW.
        #theta = numpy.arange(-math.pi+0.5*tbinw, pimage.shape[1]*tbinw, tbinw)
    #FIXME: check if theta has correct dimension if passed in

    # If the number of bins in the cartesian image is not specified, set it to
    # be the same as the number of radial bins in the polar image
    if xbins == None:
        xbins = pimage.shape[0]

    if zbins == None:
        zbins = pimage.shape[0]
    
    print 'bins', xbins, zbins
    print 'r', r
    print 'thta',theta
    print 'pimage',pimage

    
    # x and y both go from -rmax to +rmax

    # x = (numpy.arange(xbins) * xbinw) - rmax
    # z = (numpy.arange(zbins) * zbinw) - rmax

    # x = numpy.indices(-rmax, rmax, xbins)
    # z = numpy.indices(-rmax, rmax, zbins)

    # Set up the interpolation
    interp = spint.RectBivariateSpline(r, theta, pimage)

    # x, z = numpy.ogrid[0:xbins, 0:zbins]
    # x = (x * xbinw) - rmax
    # z = (z * zbinw) - rmax

    rbinw = r[1] - r[0] # Assume equally spaced radial bins
    rmax = r[-1] + (0.5 * rbinw)
    xbinw = 2.0 * rmax / xbins
    zbinw = 2.0 * rmax / zbins

    #x, z = numpy.ogrid[-rmaxi:rmaxi:xbinw*1j, -rmaxi:rmaxi:zbinw*1j]
    x, z = numpy.ogrid[0:xbins, 0:zbins]
    x = ((x + 0.5) * xbinw) - rmax
    z = ((z + 0.5) * zbinw) - rmax

    _r = numpy.sqrt(numpy.square(x) + numpy.square(z))
    _theta = numpy.arctan2(x, z)
    print xbinw, zbinw, rmax
    print 'x', x
    print 'z', z
    print '_r', _r
    print '_theta', _theta
    cimage = interp.ev(_r.ravel(), _theta.ravel()).reshape(_r.shape)
    print 'cimage', cimage
    print x,z 
    return x.flatten(), z.flatten(), cimage


from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


def polar2cartesian(r, theta, vals, x, y, order=3):

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X * X + Y * Y)
    new_t = np.arctan2(X, Y)

    # Use interpolation to connect array indices and coordinates 
    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(theta, np.arange(len(theta)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_ir, new_it]),
                            order=order).reshape(new_r.shape)

def cart2pol(image, x=None, y=None, radial_bins=256, 
             angular_bins=256, centre=None, rmax=None): 
    
    if x == None:
        # Note: these are values at bin centre
        x = numpy.arange(image.shape[0]) + 0.5

    if y == None:
        # Note: these are values at bin centre
        y = numpy.arange(image.shape[1]) + 0.5

    # centre is the value of the centre coordinate, rather than the pixel
    # number 
    if centre == None:
        xc = 0.5 * (x[0] + x[-1])
        yc = 0.5 * (y[0] + y[-1])
    else:
        xc = centre[0]
        yc = centre[1]

    x = x - xc
    y = y - yc

    # Calculate minimum distance from centre to edge of image - this
    # determines the maximum radius in the polar image. Specifically, rmax is
    # defined as the value of r at the centre of the outermost radial pixel.
    xsize = min(x[0], x[-1])
    ysize = min(y[0], y[-1])
    max_rad = min(xsize, ysize)

    if rmax == None:
        rmax = max_rad
    elif rmax > max_rad:
        raise ValueError

    # Polar image bin widths
    rbinw = rmax / (radial_bins - 0.5)
    tbinw = (2.0 * math.pi) / angular_bins
    
    

    # Calculate image in polar representation
    pimage = interp.ev(_x.ravel(), _y.ravel()).reshape(_x.shape)
    return r.flatten(), theta.flatten(), pimage

def cartesian2polar(x, y, grid, r, theta, order=3):

    R, T = np.meshgrid(r, theta)

    new_x = R * np.sin(T)
    new_y = R * np.cos(T)

    ix = interp1d(x, np.arange(len(x)))
    iy = interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    
#    new_ir[new_r.ravel() > r.max()] = len(r)-1
#    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_iy, new_iy]),
                            order=order).reshape(new_x.shape)

