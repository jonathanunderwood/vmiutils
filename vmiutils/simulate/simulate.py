import vmiutils as vmi
import vmiutils.ChoNa as chona
import numpy
import numpy.polynomial.legendre as legendre

def make_distribution(func, bins=128):
    """Creates a grided representation of function 'func' which is bins
    large in each dimension.
    """
    d = numpy.fromfunction(func, (dim, dim))

    return d

def make_image(dist):
    """Given a gridded representation of a function, produce a
    corresponding VMI simulated image using the forward Abel transform
    via the Cho and Na area matrix method.
    """
    pass

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

    def calc_dist(i, j):
        """Returns probability at position (x, y).
        """
        x = float (i)
        y = float (j)
        r = math.sqrt (x * x + y * y)
        theta = math.arccos (y / r)
        a = (r - self.r0) / self.sigma
        rad = math.exp (a * a)
        

        
