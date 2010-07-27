

class PBasexFit():
    def __init__(self, image, matrix, section='whole'):
        # TODO: check that image and matrix are instances of the correct class
        if (image.Rbins != matrix.Rbins) or (image.Thetabins != matrix.Thetabins):
            logger.error("Image and matrix do not have compatible dimensions")
            raise ValueError
        
        kdim = matrix.kmax + 1
        ldim = matrix.lmax + 1

        if section == 'whole':
            # Fit the whole image
            mtx = matrix.matrix.reshape((kdim * ldim, matrix.Rbins * matrix.Thetabins)) 
            img = image.image.reshape(matrix.Rbins * matrix.Thetabins)
        elif section == 'negative':
            # Fit only the part of the image in the region Theta = -Pi..0
            if _odd(matrix.Thetabins):
                endTheta = Thetabins / 2
            else:
                endTheta = (Thetabins / 2) - 1
            halfThetabins = endTheta + 1
            mtx = matrix.matrix[:, :, :, 0:endTheta]
            mtx = mtx.reshape((kdim * ldim, matrix.Rbins * halfThetabins))
            img = image.image[:, 0:endTheta]
            img = img.reshape(matrix.Rbins * halfThetabins)
        elif section == 'positive':
            # Fit only the part of the image in the region Theta = 0..Pi
            startTheta = matrix.Thetabins / 2 # Correct for both even and odd Thetabins
            endtheta = matrix.Thetabins - 1
            halfThetabins = matrix.Thetabins - startTheta 
            mtx = matrix.matrix[:, :, :, startTheta:endTheta]
            mtx = mtx.reshape((kdim * ldim, matrix.Rbins * halfThetabins))
            img = image.image[:, startTheta:endTheta]
            img = img.reshape(matrix.Rbins * halfThetabins)
        else:
            raise NotImplementedError
        
        coef, resids, rank, s = numpy.lstsq(mtx, img)
        coef = coef.reshape((kdim, ldim))

    
## How to reshape, reminder
# kdim = kmax + 1
# ldim = lmax + 1
# self.matrix = mtx.reshape((kdim * ldim, Rbins * Thetabins)) 
