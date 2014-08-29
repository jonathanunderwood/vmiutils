import logging
import numpy

logger = logging.getLogger('vmiutils.landweber')


class __NullHandler(logging.Handler):

    def emit(self, record):
        pass

__null_handler = __NullHandler()
logger.addHandler(__null_handler)


def projected_landweber(A, b, xguess=None, reduced_tau=1.0,
                        tolerance=1.0e-4, max_iterations=500,
                        filter_func=None, extra_args=(), extra_kwargs={}):
    """Perform projected Landweber iteration to find a solution to
    Ax=b. See 'Introduction to Inverse Problems in Imaging' by M
    Bertero and P Boccacci for a good introduction.

    Inputs
    ------
    A: matrix (required)

    b: vector (required)
    
    xguess: Initial guess at x from which iteration is started. If not
    specified, the iteration starts from x=0.

    reduced_tau: value of the relaxation (or gain) parameter
    multiplied by the squared norm of A. This must lie in the range
    0..2. Before iteration tau is calculated by dividing this by the
    square norm of A. If not specified, takes the value of 1.0.

    tolerance: the value used to terminate iteration. Iteration is
    terminated using a simple discrepancy calculation. The simulated
    vector b_k is calculated from the current x vector, x_k, from
    Ax_k=b_k. The norm of the discrepancy vector (i.e. b-b_k) is then
    calculated and normalized to the norm of b. If this value is less
    than tolerance, iteration is terminated.

    max_iterations: specifies the maximum number of iterations allowed
    before returning.

    filter_func: specifies a projection function to be called on each
    iteration. This filter function is called as:

    filter_func(x, *extra_args, **extra_kwargs)

    Returns
    -------
    x: the final vector x approximating the solution to Ax=b.
    
    """

    if A.shape[0] != b.shape[0]:
        print A.shape
        print b.shape
        logger.error('A and b have incompatible shapes')
        raise ValueError('A and b have incompatible shapes')

    if xguess is None:
        x = numpy.zeros(A.shape[1])
    elif xguess.shape[0] != A.shape[1]:
        logger.error('xguess has incompatible shapes with A and b')
        raise ValueError('xguess has incompatible shapes with A and b')
    else:
        x = xguess.copy()

    if reduced_tau < 0 or reduced_tau > 2:
        logger.error('reduced_tau does not lie between 0 and 2')
        raise ValueError('reduced_tau does not lie between 0 and 2')

    Anorm = numpy.linalg.norm(A)
    tau = reduced_tau / (Anorm * Anorm)

    AT = A.T
    ATb = numpy.dot(AT, b)
    ATA = numpy.dot(AT, A)

    bnorm = numpy.linalg.norm(b)
    bb = numpy.dot(A, x)

    for i in xrange(max_iterations):
        x = x + tau * (ATb - numpy.dot(ATA, x))
        if filter_func is not None:
            filter_func(x, *extra_args, **extra_kwargs)

        bb_new = numpy.dot(A, x)
        delta = numpy.linalg.norm(bb_new - bb) / bnorm

        if delta < tolerance:
            return x

        bb = bb_new

    logger.warning('max_iterations reached')
    return x
