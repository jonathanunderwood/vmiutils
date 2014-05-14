import vmiutils as vmi
import vmiutils.ChoNa as chona
import numpy
import numpy.polynomial.legendre as legendre

class NewtonSphere(Object):
    """Class to describe a single newton sphere definition for a particle
    group. The kinetic energy is assumed to be described by a Gaussian
    distribution centred at r0 and with a width of sigma. The angular
    distribution by an expansion in Legendre polynomials.
    """
    def __init__(self, r0, sigma, beta):
        self.r0 = r0
        self.sigma = sigma
        self.beta = numpy.asarray(beta)
        self.lmax = self.beta.shape

    def _calc_dist(i, j):
        """Returns probability at position (i, j).
        """
        x = float (i)
        y = float (j)
        r = math.sqrt (x * x + y * y)
        cos_theta = y / r
        a = (r - self.r0) / self.sigma
        rad = math.exp (a * a)
        ang = legendre.legval(cos_theta, beta)

        return rad * ang
        
    def cartesian_distribution(self, bins):
        """Returns a cartesian representation of this Newton sphere (i.e. a
        section through it). In other words, this is the modelled
        distribution, rather than the corresponding VMI image.
        """
        d = numpy.from_function(_cart_dist, (bins, bins))
        return vmi.CartesianImage(image=d)

    def vmi_image(self, bins):
        """Returns a simulated VMI image for this distribution. 
        """
        d = numpy.from_function(_cart_dist, (bins, bins))
        dist = vmi.CartesianImage(image=d)
        vmi = vmi.CartesianImage(image='empty', xbins=bins, ybins=bins)
        
        dim = max(dist.get_quadrant(i).shape[0] for i in xrange(4))

        s = chona.area_matrix(dim)

        for i in xrange(4):
            quadrant = dist.get_quadrant(i)
            dim = quadrant.shape[0]
            vmi.set_quadrant(i, numpy.dot(s, quadrant))

        return vmi


if __name__ == "__main__":
    import matplotlib.cm as cm
    import matplotlib.plot as plt

    ions = NewtonSphere(64.0, 10.0, [1.0, 2.0/math.sqrt(5.0)])

    dist = ions.cartesian_distribution(128)
    vmi = ions.vmi_image(128)


    plt.figure(1)

    plt.subplot(221)
    plt.imshow(dist.image, cmap=cm.gist_heat, origin='lower')
    
    plt.subplot(222)
    plt.imshow(dist.image, cmap=cm.spectral, origin='lower')

    plt.subplot(223)
    plt.imshow(vmi.image, cmap=cm.gist_heat, origin='lower')

    plt.subplot(224)
    plt.imshow(vmi.image, cmap=cm.spectral, origin='lower')

    plt.show()
