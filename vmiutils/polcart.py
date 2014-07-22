import numpy
import scipy.ndimage
import logging

logger = logging.getLogger('vmiutils.polcart')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


def __pol2cart(out_coord, xbw, ybw, rmax, rbw, thetabw):
    ix, iy = out_coord
    x = ix * xbw - rmax
    y = iy * ybw - rmax
    r = numpy.sqrt(x * x + y * y)
    t = numpy.arctan2(x, y)
    ir = r / rbw
    it = (t + numpy.pi) / thetabw
    return ir, it


def __cart2pol(out_coord, rbw, thetabw, xbw, ybw, xc, yc, x0, y0):
    ir, it = out_coord
    r = ir * rbw
    t = it * thetabw - numpy.pi
    x = r * numpy.sin(t)
    y = r * numpy.cos(t)
    ix = (x + xc - x0) / xbw
    iy = (y + yc - y0) / ybw
    return ix, iy


def cart2pol(image, x=None, y=None, centre=None,
             radial_bins=256, angular_bins=256, rmax=None, order=3):
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
        x = numpy.arange(float(image.shape[0]))

    if y is None:
        y = numpy.arange(float(image.shape[1]))

    if centre is None:
        xc = 0.5 * (x[0] + x[-1])
        yc = 0.5 * (y[0] + y[-1])
    else:
        xc = centre[0]
        yc = centre[1]

    # Cartesian image bin widths - assume regularly spaced
    xbw = x[1] - x[0]
    ybw = y[1] - y[0]

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

    pimage = scipy.ndimage.geometric_transform(
        image, __cart2pol, order=order,
        extra_arguments=(rbw, thetabw, xbw, ybw, xc, yc, x[0], y[0]),
        output_shape=(radial_bins, angular_bins)
    )

    r = numpy.linspace(0.0, rmax, radial_bins)
    t = numpy.linspace(-numpy.pi, numpy.pi, angular_bins)

    return r, t, pimage


def pol2cart(image, r=None, theta=None, xbins=None, ybins=None, order=3):
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
        r = numpy.arange(float(image.shape[0]))

    rbw = r[1] - r[0]  # Assume equally spaced

    if theta is None:
        theta = numpy.linspace(-numpy.pi, numpy.pi, tpts)

    tpts = image.shape[1]
    thetabw = theta[1] - theta[0]  # Assume equally spaced

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

if __name__ == '__main__':
    import matplotlib.pyplot as plot
#    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import mpl_toolkits.axes_grid1 as axes_grid1

    # Set up a very simple cartesian image - strategy will be to
    # round-trip the image to polar coordinates and back again and
    # examine it
    x = numpy.arange(10)
    y = numpy.arange(10)
    a = numpy.zeros((10, 10))
    a[5, 5] = 1.0
    a[4, 4] = 1.0
    a[4, 5] = 1.0

    # Convert to polar coordinates
    r, theta, b = cart2pol(
        x=x, y=y, image=a, order=5, radial_bins=100, angular_bins=100)

    # Convert back to cartesian coordinates
    x2, y2, c = pol2cart(
        r=r, theta=theta, image=b.clip(0.0), xbins=10, ybins=10, order=5)

    # Now plot and check
    fig = plot.figure()
    fig.set_tight_layout(True)

    # Plot the initial cartesian data with pcolormesh
    xbw = x[1] - x[0]
    ybw = y[1] - y[0]
    x_aug = numpy.append(x, x[-1] + xbw)
    y_aug = numpy.append(y, y[-1] + ybw)

    ax = plot.subplot2grid((1, 4), (0, 0), aspect=1.0)
    im = ax.pcolormesh(x_aug, y_aug, a.T)
    ax.set_xlim((x_aug[0], x_aug[-1]))
    ax.set_ylim((y_aug[0], y_aug[-1]))
    ax.set_title('Original data\n(pcolormesh)')
    ax.grid()
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # Plot the polar data directly using pcolormesh in two ways -
    # first by using a polar projection, and secondly manually
    # transforming the data grid -pcolormesh doesn't require regularly
    # spaced data (unlike imshow)
    r_aug = numpy.append(r, r[-1] + r[1] - r[0])
    theta_aug = numpy.append(theta, theta[-1] + theta[1] - theta[0])

    # Plot using polar projection
    ax = plot.subplot2grid((1, 4), (0, 1), projection="polar", aspect=1.)
    im = ax.pcolormesh(theta, r, b)
    ax.grid()
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_title('Polar data\n(pcolormesh/polar\nprojection)')
    # Unfortunately the colorbar seems broken in matplotlib 1.3.1 - bug?
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # Plot using manual conversion
    ax = plot.subplot2grid((1, 4), (0, 2),  aspect=1.)
    rg, tg = numpy.meshgrid(r_aug, theta_aug)
    xx = rg * numpy.sin(tg)
    yy = rg * numpy.cos(tg)
    im = ax.pcolormesh(xx, yy, b.transpose())
    rmax = r_aug.max()
    ax.axis([-rmax, rmax, -rmax, rmax])
    ax.set_title('Polar data\n(pcolormesh/manual\npolar conversion)')
    ax.grid()
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # Now plot the final data after the roundtrip
    x2bw = x2[1] - x2[0]
    y2bw = y2[1] - y2[0]
    x2_aug = numpy.append(x2, x2[-1] + x2bw)
    y2_aug = numpy.append(y2, y2[-1] + y2bw)

    ax = plot.subplot2grid((1, 4), (0, 3), aspect=1.0)
    im = ax.pcolormesh(x2_aug, y2_aug, c.T)
    ax.set_title('Final data\n(pcolormesh)')
    ax.axis([x2_aug[0], x2_aug[-1], y2_aug[0], y2_aug[-1]])
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plot.show()
