# TODO: use numpy.linsapce rather than numpy.arange where appropriate

import math
import numpy
import scipy.interpolate as spint

def cartesian_to_polar(cimage, x=None, y=None, radial_bins=256, 
                       angular_bins=256, centre=None, max_radius=None): 
    """ Convert an image on a regularly spaced cartesian grid into a regular
    spaced grid in polar coordinates using interpolation.
    
    x and y contain the x and y coordinates corresponding to each
    bin centre. If either of these are none, bin widths are used.

    radial_bins and angular_bins define the number of bins in the polar
    representation of the image.

    centre is a tuple containing the x and y coordinates of the image centre
    (does not need to be an integer). If this is set to None the midpoint x
    and y coordinates are used.

    max_radius defines the maximum dist
    """
    print angular_bins
    if x == None or y == None:
        xdim = cimage.shape[0]
        ydim = cimage.shape[1]

        if centre == None:
            xc = xdim * 0.5
            yc = ydim * 0.5 
        else:
            xc = centre[0]
            yc = centre[1]

        # Note: these are values at bin centre
        x = numpy.arange(xdim) + 0.5 - xc
        y = numpy.arange(ydim) + 0.5 - yc

        xmax = x[-1]
        xmin = x[0]
        ymax = y[-1]
        ymin = y[0]

    else:
        xmax = x[-1]
        xmin = x[0]
        ymax = y[-1]
        ymin = y[0]

        if centre == None:
            xc = 0.5 * (xmax - xmin)
            yc = 0.5 * (ymax - ymin)

    # Calculate bin widths in cartesian image. We assume equally spaced bin
    # widths.
    xbinw = x[1] - x[0]
    ybinw = y[1] - y[0]

    # Calculate minimum distance from centre to edge of image - this
    # determines the maximum radius in the polar image 
    xsize = min (xmax + (0.5 * xbinw) - xc, xc - xmin - (0.5 * xbinw))
    ysize = min (ymax + (0.5 * ybinw) - yc, yc - ymin - (0.5 * ybinw))
    max_rad = math.sqrt(xsize**2 + ysize**2)

    if max_radius == None:
        max_radius = max_rad
    elif max_radius > max_rad:
        raise ValueError

    # Set up interpolation - cubic spline with no smoothing by default 
    interp = spint.RectBivariateSpline(x, y, cimage)

    # Polar image bin widths
    theta_bin_width = (2.0 * math.pi) / (angular_bins - 1.0)
    radial_bin_width = max_radius / (radial_bins - 1.0)

    # Calculate polar image values - use vectorization for efficiency
    # Because we broadcast when using a ufunc we can use an ogrid here rather
    # thanan mgrid
    r, theta = numpy.ogrid[0:radial_bins, 0:angular_bins]
    theta = (theta + 0.5) * theta_bin_width
    r = (r + 0.5) * radial_bin_width

    # This creates a vectorized function. We may gain efficiency by also
    # passing in the interp object?
    polar_pix_val = numpy.frompyfunc(
        lambda r, theta: interp.ev(r * math.sin(theta), r * math.cos(theta)),
        2, 1)

    pimage = polar_pix_val(r, theta)
    print 'here'
    print pimage
    return r.flatten(), theta.flatten(), pimage

    # Calculate polar image values - non-vectorized version
    # self.pimage = numpy.empty((radial_bins, angular_bins))
    # for r in xrange(radial_bins):
    #     R = (r + 0.5) * radial_bin_width;
    #     for t in xrange(angular_bins):
    #         theta = (t + 0.5) * theta_bin_width
    #         x = R * math.sin(theta)
    #         y = R * math.cos(theta)
    #         self.pimage[r, t] = interp.ev(x, y)


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
    print r.shape
    if r == None:
        r = numpy.arange(pimage.shape[0]) + 0.5
        
    if theta == None:
        theta = numpy.arange(pimage.shape[1]) + 0.5

    

    # If the number of bins in the cartesian image is not specified, set it to
    # be the same as the number of radial bins in the polar image
    if xbins == None:
        xbins = pimage.shape[0]

    if zbins == None:
        zbins = pimage.shape[0]

    rbinw = r[1] - r[0] # Assume equally spaced radial bins
    rmax = r[-1] + (0.5 * rbinw)

    xbinw = 2.0 * rmax / (xbins - 1)
    zbinw = 2.0 * rmax / (zbins - 1)

    x = (numpy.arange(xbins) * xbinw) - rmax
    z = (numpy.arange(zbins) * zbinw) - rmax

    # Set up the interpolation
    interp = spint.RectBivariateSpline(r, theta, pimage)

    # Non-vectorized
    # ix = iy = 0
    # for xx in x:
    #     ix += 1
    #     for z in z:
    #         iz += 1
    #         rr = math.sqrt(xx**2 + zz**2)
    #         if rr > rmax:
    #             cimage[ix][iy] = 0.0
    #         else:
    #             tt = atan2(xx, zz)
    #             cimage[ix][iy] = interp.ev(rr, tt)


    def get_pix_val(x, z):
        r = math.sqrt(x**2 + z**2) # FIXME: Use numpy.sqrt here?
        if r > rmax:
            return 0.0
        else:
            return interp.ev(r, math.atan2(x,z)) # FIXME: Use numpy.atan2 here?
        
    get_pix_val_v = numpy.frompyfunc(get_pix_val, 2, 1)
    x, z = numpy.ogrid[0:xbins, 0:zbins]
    x = (x * xbinw) - rmax
    z = (z * zbinw) - rmax

    cimage = get_pix_val_v(x, z)

    # for i in xrange(cimage.shape[0]):
    #     for j in xrange(cimage.shape[1]):
    #         print i, j, cimage[i,j]

    print type(cimage[3,129])
    return x.flatten(), z.flatten(), cimage
