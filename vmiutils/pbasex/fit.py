# TODO: Many functions don't honour oddl presently

import logging
import numpy.linalg
import vmiutils as vmi
import matrix as pbm
from _fit import *

# Note: the way the matrix is calculated and stored is such that the indices
# are in the order matrix[k, l, Rbin, Thetabin]

class PbasexFit():
    def __init__(self):
        self.coefs = None
        self.kmax = None
        self.lmax = None
        self.oddl = None
        self.sigma = None
        self.rkstep = None
        self.rfactor = None
        self.rmax = None
        self.__attribs = ['coefs', 'kmax', 'lmax', 'oddl', 'sigma',
                          'rkstep', 'rfactor', 'rmax']

    def fit_data(self, image, matrix, section='whole', lmax=None, oddl=None):
        if not isinstance(image, vmi.PolarImage):
            logger.error("image is not an instance of PolarImage")
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
        elif oddl is True and matrix.oddl is True:
            mtx = matrix.matrix
        elif oddl is False and matrix.oddl is True:
            mtx = matrix.matrix[:, 0:matrix.lmax + 1:2, :, :]
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

        coef = coef.reshape((kdim, ldim))

        # If oddl is False we need to expand the coef array to include the odd
        # l values and set them to zero, such that coef is indexed as coef[k,
        # l] rather than coef[k, l/2]. Then all other functions don't have to
        # care about oddl.
        if oddl is False:
            self.coef = numpy.zeros((kdim, lmax + 1))
            for l in xrange(0, lmax + 1, 2):
                self.coef[:, l] = coef[:, l / 2]
        else:
            self.coef = coef

        self.kmax = matrix.kmax
        self.lmax = lmax
        self.oddl = oddl
        self.sigma = matrix.sigma

        # rmax is the maximum radial bin number we can sensibly consider
        # i.e. without extrapolating outside the input data
        self.rmax = Rbins - 1

        # rkstep holds the spacing between the centres of adjacent Gaussian
        # radial basis functions
        self.rkstep = float(Rbins) / kdim

        # rfactor holds the scaling factor to convert from radial bin number
        # in the polar image to actual position in the original image 
        self.rscale = image.R[1] - image.R[0]

        self.fit_done = True

    def calc_radial_spectrum(self, rbins=500, rmax=None):
        """Calculate a raidal spectrum from the parameters of a fit. Returns a
        tuple (r, intensity) containing the r values and the corresponding
        intensities. 

        rbins determines the number of points in the returned spectrum.

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

        spec = calc_spectrum2(float(rmax), rbins, self.coef, self.kmax, 
                              self.rkstep, self.sigma)
        r = numpy.linspace(0.0, rmax * self.rscale, rbins)

        return r, spec

    def calc_distribution(self, rbins=512, thetabins=512, rmax=None):
        if self.fit_done is False:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError
        
        dist = calc_distribution2(rmax, rbins, thetabins, self.coef, self.kmax,
                                  self.rkstep, self.sigma, self.lmax)
        r = numpy.linspace(0.0, rmax * self.rscale, rbins)
        theta = numpy.linspace(-numpy.pi, numpy.pi, thetabins)

        return r, theta, dist

    def cartesian_distribution(self, bins=500, rmax=None):
        if self.fit_done is False:
            logger.error('no fit done')
            raise AttributeError
        
        if rmax is None:
            rmax = self.rmax
        elif rmax > self.rmax:
            logger.error('rmax exceeds that of original data')
            raise ValueError
        
        dist = cartesian_distribution(rmax, bins, self.coef, self.kmax,
                                      self.rkstep, self.sigma, self.lmax)

        x = numpy.linspace(-rmax, rmax, bins)
        y = numpy.linspace(-rmax, rmax, bins)

        return x, y, dist

    def beta_coefficients(self, rbins=500, rmax=None):
        if rmax == None:
            rmax = self.rmax

        beta = beta_coeffs(rmax, rbins, self.coef, self.kmax, 
                           self.rkstep, self.sigma, self.lmax)
        r = numpy.linspace(0.0, rmax * self.rscale, rbins)
        return r, beta

    def dump(self, file):
        fd = open(file, 'r')
        
        try:
            for object in self.__attribs:
                setattr(self, object, pickle.load(fd))
        finally:
            fd.close()

