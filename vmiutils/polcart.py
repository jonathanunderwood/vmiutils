import numpy
import scipy.ndimage

def __pol2cart(out_coord, xbw, ybw, rmax, rbw, thetabw):
    ix, iy = out_coord
    x = ix * xbw - rmax
    y = iy * ybw - rmax
    r = numpy.sqrt(x * x + y * y)
    t = numpy.arctan2(x, y)
    ir = r / rbw
    it = (t + numpy.pi) / thetabw
    return ir, it

def __cart2pol(out_coord, rbw, thetabw, xbw, ybw, xc, yc):
    ir, it = out_coord
    r = ir * rbw
    t = it * thetabw - numpy.pi
    x = r * numpy.sin(t)
    y = r * numpy.cos(t)
    ix = (x + xc) / xbw
    iy = (y + yc) / ybw
    return ix, iy

def cart2pol(image, x=None, y=None, centre=None, 
             radial_bins=256, angular_bins=256, rmax=None, order=5):
    """ Convert an image on a regularly spaced cartesian grid into a regular
    spaced grid in polar coordinates using interpolation.
    
    x and y contain the x and y coordinates corresponding to each bin. If
    either of these are none, unit bin widths are assumed.

    radial_bins and angular_bins define the number of bins in the polar
    representation of the image.

    centre is a tuple containing the x and y coordinates of the image centre
    (does not need to be an integer). If this is set to None the midpoint x
    and y coordinates are used.

    rmax defines the maximum radius from the image centre to consider. If rmax
    is None, rmax is set to be the smallest of the distances from the image
    centre to the edge of the cartesian grid.

    We employ the convention that the angle (theta) is that between the second
    axis (y-axis) and the position vector and that it lies in the range
    [-pi,pi].

    A tuple (r, theta, pimage) is returned, with r and theta containing the
    coordinates of each bin in the polar image pimage.

    order specifies the interpolation order.
    """
    
    if x is None:
        x = numpy.arange(image.shape[0])

    if y is None:
        y = numpy.arange(image.shape[1])

    if centre is None:
        xc = 0.5 * (x[0] + x[-1])
        yc = 0.5 * (y[0] + y[-1])
    else:
        xc = centre[0]
        yc = centre[1]

    # Calculate minimum distance from centre to edge of image - this
    # determines the maximum radius in the polar image.
    xsize = min(abs(x[0] - xc), x[-1] - xc)
    ysize = min(abs(y[0] - yc), y[-1] - yc)
    max_rad = min(xsize, ysize)

    if rmax is None:
        rmax = max_rad
    elif rmax > max_rad:
        raise ValueError

    # Polar image bin widths
    rbw = rmax / (radial_bins - 1)
    thetabw = 2.0 * numpy.pi / (angular_bins - 1)

    # Cartesian image bin widths - assume regularly spaced
    xbw = x[1] - x[0]
    ybw = y[1] - y[0]

    pimage = scipy.ndimage.geometric_transform(
        image, __cart2pol, order=order,
        extra_arguments=(rbw, thetabw, xbw, ybw, xc, yc),
        output_shape=(radial_bins, angular_bins)
        )

    r = numpy.linspace(0.0, rmax, radial_bins)
    t = numpy.linspace(-numpy.pi, numpy.pi, angular_bins)

    return r, t, pimage

def pol2cart(image, r=None, theta=None, xbins=None, ybins=None, order=5):
    """ Convert an image on a regularly spaced polar grid into a regular
    spaced grid in cartesian coordinates using interpolation.
    
    r and theta contain the coordinates corresponding to each bin in image. If
    either of these are none, unit bin widths are assumed.

    xbins and ybins define the number of bins in the returned cartesian
    representation of the image.

    We employ the convention that the angle (theta) is that between the second
    axis (y-axis) and the position vector and that it lies in the range
    [-pi,pi].

    A tuple (x, y, cimage) is returned, with x and y containing the
    coordinates of each bin in the cartesian image cimage.

    order specifies the interpolation order.
    """

    if r is None:
        r = numpy.arange(image.shape[0])

    rbw = r[1] - r[0] # Assume equally spaced

    if theta is None:
        theta = numpy.linspace(-numpy.pi, numpy.pi, tpts)

    tpts = image.shape[1]
    thetabw = theta[1] - theta[0] # Assume equally spaced

    # If the number of bins in the cartesian image is not specified, set it to
    # be the same as the number of radial bins in the polar image
    if xbins is None:
        xbins = image.shape[0]

    if ybins is None:
        ybins = image.shape[0]

    rmax = r[-1]
    xbw = 2.0 * rmax / (xbins - 1)
    ybw = 2.0 * rmax / (ybins - 1)

    cimage = scipy.ndimage.geometric_transform(
        image, __pol2cart, order=order, 
        extra_arguments=(xbw, ybw, rmax, rbw, thetabw),
        output_shape=(xbins, ybins)
        )

    x = numpy.linspace(-rmax, rmax, xbins)
    y = numpy.linspace(-rmax, rmax, ybins)

    return x, y, cimage

    
