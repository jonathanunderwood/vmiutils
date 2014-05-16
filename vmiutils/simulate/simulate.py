import vmiutils as vmi
import vmiutils.ChoNa as chona
import numpy
import numpy.polynomial.legendre as legendre
import math

class NewtonSphere(object):
    """Class to describe a single newton sphere definition for a particle
    group. The kinetic energy is assumed to be described by a Gaussian
    distribution centred at r0 and with a width of sigma. The angular
    distribution by an expansion in Legendre polynomials.
    """
    def __init__(self, r0, sigma, beta):
        self.r0 = r0
        self.sigma = sigma
        self.beta = numpy.asarray(beta)

    def _cart_dist(self, x, y, **kwargs):
        """Returns probabilities for mesh grid (x, y). If the kwargs dict has
        a key named centre with a corresponding tuple (xc, yc) as the
        value, we'll use that as the image centre, otherwise assumes
        the centre of the image is at the centre of the grid.
        """
        centre = kwargs.get('centre', None)

        if centre is None:
            xc = x.shape[0] / 2.0
            yc = y.shape[0] / 2.0
        else:
            xc = centre[0]
            yc = centre[1]

        # Since x and y are pixel indices, we need to calculate the
        # distance from the *centre* of the pixel (i.e. [x+0.5,
        # y+0.5]) from the image centre
        xx = (x + 0.5) - xc
        yy = (y + 0.5) - yc

        r = numpy.sqrt (xx**2 + yy**2)

        # Note that the following causes RuntimeWarnings:
        # cos_theta = numpy.where(r > 0.0, y / r, 0.0)
        # because the y/r is evaluated before the where.
        # See:
        # http://mail.scipy.org/pipermail/numpy-discussion/2013-January/064988.html
        cos_theta = numpy.zeros_like(yy)
        numpy.divide (yy, r, where=(r > 0.0), out=cos_theta)

        a = (r - self.r0) / self.sigma
        rad = numpy.exp (-0.5 * a**2) / (self.sigma * math.sqrt(2.0 * math.pi))

        ang = legendre.legval(cos_theta, self.beta)

        return rad * ang
        
    def cartesian_distribution(self, bins, centre=None, resample=False):
        """Returns a cartesian representation of this Newton sphere (i.e. a
        section through it). In other words, this is the modelled
        distribution, rather than the corresponding VMI image.

        The optional 'centre' argument specifies where the image is
        centred on the grid (default is the grid centre).

        resample specifies whether to resample the image when setting
        the centre so as to ensure the image centre lies on the lower
        left corner of a pixel, such that the quadrants are
        exact. This is done if resample is True. If resample is False
        (the default) the image isn't resampled, and the quadrants are
        taken about the nearest pixel to the image centre.
        """
        if (bins % 2) != 0:
            logger.error ('bins needs to be an even integer')
            raise ValueError('bins needs to be an even integer')

        d = numpy.fromfunction(self._cart_dist, (bins, bins), centre=centre)

        return vmi.CartesianImage(image=d, centre=centre, resample=resample)

    def vmi_image(self, bins, centre=None, resample=False):
        """Returns a simulated VMI image for this distribution as a cartesian
        grid of binsxbins size. bins should be an even number.

        The optional 'centre' argument specifies where the image is
        centred on the grid (default is the grid centre).

        resample specifies whether to resample the image when setting
        the centre so as to ensure the image centre lies on the lower
        left corner of a pixel, such that the quadrants are exact when
        the VMI image is calculated. This is done if resample is
        True. If resample is False (the default) the image isn't
        resampled, and the quadrants are taken about the nearest pixel
        to the image centre, and some differences between the
        quadrants in the calculated image will occur.
        """
        if (bins % 2) != 0:
            logger.error ('bins needs to be an even integer')
            raise ValueError('bins needs to be an even integer')

        d = numpy.fromfunction(self._cart_dist, (bins, bins),
                               centre=centre)
        dist = vmi.CartesianImage(image=d, centre=centre,
                                  resample=resample)

        vmi_img = vmi.CartesianImage(image='empty',
                                     xbins=bins, ybins=bins,
                                     centre=centre,
                                     resample=resample)
        
        dim = max(dist.get_quadrant(i).shape[0] for i in xrange(4))

        s = chona.area_matrix(dim)

        for i in xrange(4):
            quadrant = dist.get_quadrant(i)
            dim = quadrant.shape[0]
            vmi_img.set_quadrant(i, numpy.dot(s[0:dim, 0:dim], quadrant))

        return vmi_img


if __name__ == "__main__":
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import math

    ions = NewtonSphere(20.0, 2.0, [1.0, 0.0, 2.0/math.sqrt(5.0)])

    dist = ions.cartesian_distribution(64, centre=(25.5, 25.5), resample=True)
    vmi_img = ions.vmi_image(64, centre=(25.5, 25.5), resample=True)

    plt.figure(1)

    plt.subplot(221)
    plt.imshow(dist.image.transpose(), cmap=cm.gist_heat, origin='lower')
    
    plt.subplot(222)
    plt.imshow(dist.image.transpose(), cmap=cm.spectral, origin='lower')

    plt.subplot(223)
    plt.imshow(vmi_img.image.transpose(), cmap=cm.gist_heat, origin='lower')

    plt.subplot(224)
    plt.imshow(vmi_img.image.transpose(), cmap=cm.spectral, origin='lower')

    plt.show()
