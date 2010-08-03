import logging
import vmiutils as vmi
import vmiutils.pbasex.matrix as pbm
from _basisfn import *

# Note: the way the matrix is calculated and stored is such that the indices
# are in the order matrix[k, l, Rbin, Thetabin]

class PbasexFit():
    def __init__(self):
        self.coefs = None
        self.kmax = None
        self.lmax = None
        self.oddl = None
        self.sigma = None
        self.rkspacing = None
        self.rfactor = None
        self.__attribs = ['coefs', 'kmax', 'lmax', 'oddl', 'sigma',
                          'rkspacing', 'rfactor']

    def fit_data(self, image, matrix, section='whole', lmax=None, oddl=None):
        if not isinstance(image, vmi.VMIPolarImage):
            logger.error("image is not an instance of VMIPolarImage")
            raise TypeError
        elif not isinstance(matrix, pbm.PbasexMatrix):
            logger.error('matrix is not an instance of PbasexMatrix')
            raise TypeError
        elif (image.Rbins is not matrix.Rbins) or (image.Thetabins is not matrix.Thetabins):
            logger.error("image and matrix do not have compatible dimensions")
            raise TypeError
        
        if oddl is True and matrix.oddl is False:
            logger.error('odd l requested, but matrix not calculated for odd l')
            raise ValueError
        elif oddl is False and matrix.oddl is True:
            mtx = mtx[:, 0:matrix.lmax + 1:2, :, :]
        elif oddl is None:
            mtx = matrix.matrix
            oddl = matrix.oddl
        
        if lmax is not None:
            if lmax > matrix.lmax:
                logger.error('requested lmax greater than that of supplied matrix')
                raise ValueError
            else:
                if oddl is True:
                    ldim = lmax + 1
                else:
                    ldim = lmax / 2 + 1
                mtx = mtx[:, 0:ldim, :, :]
        else:
            lmax = matrix.lmax
            if oddl is True:
                ldim = lmax + 1
            else:
                ldim = lmax / 2 + 1

        kdim = matrix.kmax + 1
        Thetabins = matrix.Thetabins
        Rbin = matrix.Rbins

        if section == 'whole':
            # Fit the whole image
            mtx = mtx((kdim * ldim, Rbins * Thetabins)) 
            img = image.image.reshape(Rbins * Thetabins)
        elif section == 'negative':
            # Fit only the part of the image in the region Theta = -Pi..0
            if _odd(Thetabins):
                endTheta = Thetabins / 2
            else:
                endTheta = (Thetabins / 2) - 1
            halfThetabins = endTheta + 1
            mtx = mtx[:, :, :, 0:endTheta]
            mtx = mtx.reshape((kdim * ldim, Rbins * halfThetabins))
            img = image.image[:, 0:endTheta]
            img = img.reshape(Rbins * halfThetabins)
        elif section == 'positive':
            # Fit only the part of the image in the region Theta = 0..Pi
            startTheta = Thetabins / 2 # Correct for both even and odd Thetabins
            endtheta = Thetabins - 1
            halfThetabins = Thetabins - startTheta 
            mtx = matx[:, :, :, startTheta:endTheta]
            mtx = mtx.reshape((kdim * ldim, Rbins * halfThetabins))
            img = image.image[:, startTheta:endTheta]
            img = img.reshape(Rbins * halfThetabins)
        else:
            raise NotImplementedError
        
        coef, resid, rank, s = numpy.lstsq(mtx, img)
        # TODO: do something with resid

        self.coef = coef.reshape((kdim, ldim))
        self.kmax = kmax
        self.lmax = lmax
        self.oddl = oddl
        self.sigma = matrix.sigma
        self.rkspacing = Rbins / kdim
        self.rfactor = image.rfactor
        self.fit_done = True

        def calc_radial_spectrum(self, nbins):
            if self.fit_done is False:
                logger.error('no fit done')
                raise AttributeError
            
            spec = numpy.zeros(nbins)
            r = numpy.empty(nbins)

            for i in xrange(nbins):
                r[i] = i * self.rspacing
                for k in xrange(self.kmax + 1):
                    rk = k * self.rkspacing
                    spec[i] += basisfn_radial (r[i], rk, self.sigma)

            return r, spec

        def dump(self, file):
            fd = open(file, 'r')
                    
            try:
                for object in self.__attribs:
                    setattr(self, object, pickle.load(fd))
            finally:
                fd.close()

