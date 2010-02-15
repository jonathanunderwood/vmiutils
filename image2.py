import numpy
import scipy.interpolate as spint

class VMCartesianImage():
    """ Class used to represent a VMI image stored as a cartesian
    array. Instantiation requires passing the image data stored in a numpy
    array. 
    """
    def __init__(self, image):
        self.image = image
        
    def centre_of_gravity(self):
        """ Returns thea tuple representing the coordinates corresponding to
        the centre of gravity of the image. This is not rounded to the nearest
        integers.
        """
        sum = 0.0
        xval = 0.0
        yval = 0.0
        
        for index, val in numpy.ndenumerate(self.image):
            sum += val
            # Need to add 0.5 to the index to allow for the fact that the
            # centre coordinate of each pixel is the index plus 0.5
            xval += val * (index[0] + 0.5)
            yval += val * (index[1] + 0.5)


        x, y = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]
        x += 0.5
        y += 0.5
        imsum = self.image.sum()
        xval = (x * self.image).sum() / sum
        yval = (y * self.image).sum() / sum
                           
        # The following rounding gives the centre pixel
        # return int(round(xval / sum)), int(round(yval / sum))

        # Calculating xval/sum gives the x coordinate of the centre of
        # gravity, but we need to take into account that the array indices
        # used in the calculation above are the left most values of each pixel
        # rather than the centre
        # Note: this not necessary if we add 0.5 to the indices above.
        # return (xval / sum) + 0.5, (yval / sum) + 0.5 
            return xval / sum, yval / sum 

    def polar_workspace_init(radial_bins=256, angular_bins=256,
                             max_radius=None, centre=None): 
        #if (centre == None) and self.centre == None:
            #pass # Raise an exception

        xdim = self.image.shape[0]
        ydim = self.image.shape[1]

        if centre == None:
            xc = xdim * 0.5
            yc = ydim * 0.5 
        else:
            xc = centre[0]
            yc = centre[1]

        # Calculate minimum distance from centre to edge of image - this
        # determines the maximum radius in the polar image
        xsize = min (xdim + 0.5 - xc, xc)
        ysize = min (ydim + 0.5 - yc, yc)
        max_rad = m.sqrt(xsize**2 + ysize**2)

        if max_radius == None:
            max_radius = max_rad
        elif max_radius > max_rad:
            raise ValueError
        
        # Set up interpolation - cubic spline with no smoothing by default 
        x = numpy.indices((xdim,)) + 0.5 - centre[0]
        y = numpy.indices((ydim,)) + 0.5 - centre[1]
        interp = spint.RectBivariateSpline(x, y, self.image)

        # Polar image bin widths
        theta_bin_width = (2.0 * math.pi) / (theta_bins - 1.0)
        radial_bin_width = max_radius / (radial_bins - 1.0)

        # Calculate polar image values - use vectorization for efficiency
        # Because we broadcast when using a ufunc (created by frompyfunc
        # below), we could get away with an ogrid here to save time and space?
        r, theta = numpy.mgrid[0:radial_bins, 0:angular_bins]
        theta = (theta + 0.5) * theta_bin_width
        r = (r + 0.5) * radial_bin_width

        def polar_pix_val(r, theta):
            # Should we use the numpy.sin/cos functions here for more
            # efficiency ?
            return interp.ev(r * m.sin(theta), r * m.cos(theta))

        numpy.frompyfunc(polar_pix_val, 2, 1)
        self.pimage = polar_pix_val(r, theta)

        # Calculate polar image values - non-vectorized version
        self.pimage = numpy.empty(radial_bins, angular_bins)
        for r in radial_bins:
            R = (r + 0.5) * radial_bin_width;
            for t in theta_bins:
                theta = (t + 0.5) * theta_bin_width
                x = R * sin(theta)
                y = R * cos(theta)
                self.pimage[r, t] = interp.ev(x, y)

if __name__ == "__main__":
    import sys
    import pylab

    file=sys.argv[1]

    try:
        image = numpy.loadtxt(file)
    except IOError:
        print "Could not read file", file
        sys.exit(74)

    image = image.transpose()

    cofg = centre_of_gravity(image)
    print "Centre of gravity:", cofg[0], cofg[1] 
    
    wksp = workspace_init(image, cofg, 0, 1, 2, 3)

    pylab.figure()
    pylab.imshow(wksp, origin='lower')
    pylab.show()



#     def centre_bordas(self):
#         """
#         Finds the image centre using the Bordas criterion
#         [Rev. Sci. Instrumen. Vol 67, page 2257]. This assumes that the image
#         has reflection symmetry in both the x and y axes.
#         """
#         fmin(__bordas, )

#     def __bordas()
#         # This is the function that is minimized.

#     def symmetrize(self, direction):
#         pass


    
    # def workspace_init(image, centre, *quadrants):
    #     """
    #     Set up a workspace for a subsequent inversion.

    #     centre is a tuple containing the pixel coordinates to be used as the image
    #     centre.

    #     We adopt the convention that the axis of cylindrical symmetry is
    #     associated with the second index of the numpy image array.

    #     The last argument(s) indicate the quadrants to be used for constructing
    #     the workspace. The quadrants are numbered 0-3:

    #     Quadrant 0: from centre to (xmax, ymax) [Top right]
    #     Quadrant 1: from centre to (xmax, 0)    [Bottom right]
    #     Quadrant 2: from centre to (0, 0)       [Bottom Left]
    #     Quadrant 3: from centre to (0, ymax]    [Top left]
    #     """
