import logging
import vmiutils.pbasex as pbasex

# Set up logging and create a null handler in case the application doesn't
# provide a log handler
logger = logging.getLogger('vmiutils.pbasex.matrix_detfn1')

class __NullHandler(logging.Handler):
    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


class PbasexFitDetfn1 (pbasex.PbasexFit):
    def __init__(self):
        super(PbasexFitDetfn1, self).__init__()

    def fit_data(self, image, matrix, section='whole', lmax=None, oddl=None):
        """This function simply calls the PbasexFit method, and then rescales
        sigma and rkstep appropriately.

        """
        super(PbasexFitDetfn1, self).fit_data(image, matrix, section='whole', 
                                              lmax=None, oddl=None)
        # The parent fit_data function takes the sigma and rkstep from
        # the matrix calculation (i.e. expressed in terms of the
        # matrix bins and rescales them to the image. This is fine for
        # standard PBasex, where the matrix is general. However in the
        # present case, the matrix is defined only for a specific
        # maximum radius (defined by the maximum radius used when
        # generating the detection function), and so we have to undo
        # that unneeded rescaling here.
        print self.rkstep
        self.sigma = matrix.sigma
        self.rkstep = matrix.Rbins / (matrix.kmax + 1.0)

        print self.rkstep
        print 'here'
