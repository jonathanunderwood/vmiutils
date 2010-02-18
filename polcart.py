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

    rmax defines the maximum dist
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
    print 'binw:', xbinw, ybinw

    # centre is the value of the centre coordinate, rather than the pixel
    # number 
    if centre == None:
        xc = 0.5 * (xmax - xmin + xbinw)
        yc = 0.5 * (ymax - ymin + ybinw)
    else:
        xc = centre[0]
        yc = centre[1]
    print 'centre:', xc, yc

    # Calculate minimum distance from centre to edge of image - this
    # Determine the maximum radius in the polar image 
    xsize = min(xmax + (0.5 * xbinw) - xc, abs(xmin - (0.5 * xbinw) - xc))
    ysize = min(ymax + (0.5 * ybinw) - yc, abs(ymin - (0.5 * ybinw) - yc))
    max_rad = min(xsize, ysize)
    print 'max_rad', max_rad
    print 'xmin xmax', xmin, xmax
    print 'ymin ymax', ymin, ymax
    
    print 'xsize ysize', xsize, ysize

    if rmax == None:
        rmax = max_rad
    elif rmax > max_rad:
        raise ValueError

    # Set up interpolation - cubic spline with no smoothing by default 
    interp = spint.RectBivariateSpline(x, y, cimage)

    # Polar image bin widths
    rbinw = rmax / (radial_bins - 1.0)
    tbinw = (2.0 * math.pi) / (angular_bins)

    # Calculate polar image values using interpolation, calculating all x and
    # y values in advance for efficiency
    # r, theta = numpy.ogrid[0.5*rbinw:rmax-0.5*rbinw:rbinw*1j, 
    #                        0.5*tbinw:2.0*math.pi-0.5*tbinw]

    r, theta = numpy.ogrid[0:radial_bins, 0:angular_bins]
    r = (r + 0.5) * rbinw
    theta = (theta + 0.5) * tbinw
    # theta = (theta + 0.5) * tbinw
    # r = (r + 0.5) * rbinw
    _x = r * numpy.sin(theta)
    _y = r * numpy.cos(theta)
    pimage = interp.ev(_x.ravel(), _y.ravel()).reshape(_x.shape)

    print radial_bins, angular_bins
    print 'r, rbinw:', r, rbinw
    print theta
    print pimage
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
        theta = numpy.arange(0.5, pimage.shape[1])
    #FIXME: check if theta has correct dimension if passed in

    # If the number of bins in the cartesian image is not specified, set it to
    # be the same as the number of radial bins in the polar image
    if xbins == None:
        xbins = 2 * pimage.shape[0]

    if zbins == None:
        zbins = 2 * pimage.shape[0]
    
    print 'bins', xbins, zbins
    print 'r', r
    print 'thta',theta
    print 'pimage',pimage

#    rbinw = r[1] - r[0] # Assume equally spaced radial bins
    
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
    rmaxi = r[-1] # r value at centre of bin
    xbinw = 2.0 * rmaxi / (xbins - 1)
    zbinw = 2.0 * rmaxi / (zbins - 1)

    x, z = numpy.ogrid[-rmaxi:rmaxi:xbinw*1j, -rmaxi:rmaxi:zbinw*1j]

    _r = numpy.sqrt(numpy.square(x) + numpy.square(z))
    _theta = numpy.arctan2(x, z)
    # print xbinw, zbinw, rmaxi
    # print x
    # print z
    # print _r
    # print _theta
    cimage = interp.ev(_r.ravel(), _theta.ravel()).reshape(_r.shape)

    return x.flatten(), z.flatten(), cimage
