import logging
import numpy.linalg
import vmiutils as vmi
import matrix as pbm
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
        self.rmax = None
        self.__attribs = ['coefs', 'kmax', 'lmax', 'oddl', 'sigma',
                          'rkspacing', 'rfactor', 'rmax']

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
        Rbins = matrix.Rbins

        if section == 'whole':
            # Fit the whole image
            mtx = mtx.reshape((kdim * ldim, Rbins * Thetabins)) 
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
        
        coef, resid, rank, s = numpy.linalg.lstsq(mtx.transpose(), img)
        # TODO: do something with resid

        self.coef = coef.reshape((kdim, ldim))
        self.kmax = matrix.kmax
        self.lmax = lmax
        self.oddl = oddl
        self.sigma = matrix.sigma

        # rmax is the maximum radial bin number we can sensibly consider
        # i.e. without extrapolating outside the input data
        self.rmax = Rbins - 1

        # rkspacing holds the spacing between the centres of adjacent Gaussian
        # radial basis functions
        self.rkspacing = float(Rbins) / kdim

        # rfactor holds the scaling factor to convert from radial bin number
        # in the polar image to actual position in the original image 
        self.rscale = image.R[1] - image.R[0]

        self.fit_done = True

    def calc_radial_spectrum(self, npoints=500, rmax=None):
        """Calculate a raidal spectrum from the parameters of a fit. Returns a
        tuple (r, intensity) containing the r values and the corresponding
        intensities. 

        npoints determines the number of points in the returned spectrum.

        rmax is the maximum radius to consider, i.e. the spectrum is
        claculated for r=0..rmax. Note: rmax is the desired radial value and
        not the bin number. If rmax is None (default) then the maximum radius
        in the input image is used."""

        if self.fit_done is False:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError

        return calc_spectrum(float(rmax), npoints, self.coef, self.kmax, 
                             self.rkspacing, self.sigma)

    def calc_distribution(self, rmax=None, rbins=512, thetabins=512):
        if self.fit_done is False:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError
        
        r, theta, dist = calc_distribution(rmax, rbins, thetabins, self.coef, self.kmax,
                                           self.rkspacing, self.sigma, self.lmax)
        return r, theta, dist

    def dump(self, file):
        fd = open(file, 'r')
        
        try:
            for object in self.__attribs:
                setattr(self, object, pickle.load(fd))
        finally:
            fd.close()

