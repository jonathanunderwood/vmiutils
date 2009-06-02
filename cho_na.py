# Functions relating to "Application of Abel inversion in real-time
# calculations for circularly and elliptically symmetric radiation sources",
# Y. T. Cho and S-J Na, Sci. Technol. 16 878-884 (2005).  Note that the i, j
# indices in the Cho and Na paper start from 1 rather than 0. Here we index
# from zero, so the formulas are adjusted accordingly.

import numpy
import scipy.linalg

def __P(i, j):
    """
    Calculates the Cho and Na P matrix elements assuming d = 1.
    """
    # Note that here we get away with casting i and j to float, which is
    # surprising. Note that in the future, if this is required, it is achieved
    # by jj = j + 1.0, i.astype(float) etc.
    jj = j + 1.0
    theta = numpy.arccos(i / jj)
    return numpy.where(j >= i, 
                       0.5 * (jj * jj * theta - i * i * numpy.tan(theta)), 
                       0.0)

def area_matrix(dim):
    """
    Calculate and return a numpy array containing the area matrix as defined
    by Cho and Na. The returned square array is of size dim. The array is
    normalized to unit width.
    """
    P = numpy.fromfunction(__P, (dim, dim))

    # Here we set d = 1 / dim^2 to normalize to a width of 1.  This isn't
    # really necessary. Likewise we could drop the 0.5 in __P above.
    P = numpy.multiply(P, 1.0 / (dim * dim))

    S = P.copy()
    S[0:dim - 1, :] -= P[1:dim, :]
    S[:, 1:dim] -= P[:, 0:dim - 1]
    S[0:dim - 1, 1:dim] += P[1:dim, 0:dim - 1]

    return S

def invert(image, S=None):
    """
    Invert an image stored as a 2D numpy array. Assumes that image[0, 0]
    corresponds to the first element of the area matrix etc. In other words,
    assumes that image[0, 0] is the image centre and the array contains one
    quadrant of the image. 

    image: numpy 2D array containing image to be inverted

    S: Area matrix as defined by Cho and Na. Optional - if not passed will be
    calculated as needed

    returns a 2D numpy array containing the inverted image in cylindrical
    polar coordinates.
    """

    if S == None:
        dim = image.shape[0]
        S = area_matrix(dim)
    elif S.shape[0] != image.shape[0]:
        raise ValueError # TODO: replace with own exception

    return scipy.linalg.solve(S, image)


if __name__ == "__main__":
    S = area_matrix(20)
    print S
