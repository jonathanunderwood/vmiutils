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

    # Because of the column order that linalg.solve expects, we have to
    # transpose here, and then transpose the answer back.
    im = image.transpose()

    if S == None:
        S = area_matrix(im.shape[0])
    elif S.shape[0] != im.shape[0]:
        raise ValueError

    a = scipy.linalg.solve(S, im)
    return a.transpose()

def invert_plreg(image, iterations, initial_guess=None, tau=None, S=None):

    im = image#.transpose()

    if S == None:
        S = area_matrix(im.shape[1]).transpose()
    elif S.shape[1] != im.shape[1]:
        raise ValueError

    Snorm = numpy.linalg.norm(S)
    Snorm2 = Snorm * Snorm

    if tau == None:
	tau = 1.0 / Snorm2
    elif tau <= 0 or tau >= 2 / Snorm2:
	raise ValueError

    print "tau:", tau

    St = S.transpose()
    StS = numpy.dot(St, S)

    if initial_guess == None:
        a = numpy.ones(shape=im.shape, dtype=float)
    elif initial_guess.shape != im.shape:
	raise ValueError
    else:
	a = initial_guess

    for i in xrange(im.shape[0]):
        print i
        b = numpy.dot(St, im[i])

        # Scale inital guess so that each row gives the same number of hits in
        # the corresponding synthesized image as the in the original image
        if initial_guess == None:
            norm = (numpy.dot(S, a[i])).sum()
            norm = im[i].sum() / norm
            print norm
            a[i] *= norm

        for j in xrange(iterations):
            a[i] += tau * (b - numpy.dot(StS, a[i]))
            a[i] = a[i].clip(min=0.0)

    return a

from scipy import signal

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = numpy.mgrid[-size:size+1, -sizey:sizey+1]
    g = numpy.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)





if __name__ == "__main__":
    import image
    import sys
    import pylab
    import matplotlib.cm
    
    file=sys.argv[1]

    try:
        img = numpy.loadtxt(file)
    except IOError:
        print "Could not read file", file
        sys.exit(74)
        
#    img = img.transpose()

    cofg = image.centre_of_gravity(img)

    wksp = image.workspace_init(img, cofg, 0)

    dist_simp = invert(wksp)
#    dist = invert_plreg(wksp, 1000, initial_guess=dist_simp)
#    dist = invert_plreg(wksp, 1000)

#    dist_simp=blur_image(dist_simp, 3)

    fig=pylab.figure(1)
#    pylab.subplot(1, 3, 1)
#    plt = pylab.imshow(dist[30:, 30:], cmap=matplotlib.cm.gray, origin='lower')
#    plt = pylab.imshow(dist, cmap=matplotlib.cm.gray, origin='lower')
#    fig.colorbar(plt)
    pylab.subplot(1, 2, 1)
    plt2 = pylab.imshow(dist_simp, cmap=matplotlib.cm.gray, origin='lower')
    pylab.subplot(1, 2, 2)
    plt2 = pylab.imshow(wksp, cmap=matplotlib.cm.gray, origin='lower')

    pylab.show()
