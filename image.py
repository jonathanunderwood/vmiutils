# TODO 
#
# Symmetrization - for reflection in one and two directions Deal with rotation
# of image eg. due to CCD not being aligned to laser polarization
#

import numpy

class VMI:
    def __init__(self):
        self.cofg = None
        self.centre = None
        self.quadrants = None

    def read(self, filename):
        """
        Reads a VMI data file. Assumes the data is represented as a matrix
        in the file. Lines beginning with # are ignored. Caller should check
        for IOError exception.
        """
        self.image=numpy.loadtxt(filename)
        self.datafile=filename
    
    def centre_of_gravity(self):
        """
        Finds the pixel corresponding to the centre of gravity of the VMI
        image. The centre of gravity is stored as the cofg tuple attribute,
        and also returned.
        """
        sum = 0.0
        xval = 0.0
        yval = 0.0

        for index, val in numpy.ndenumerate(self.image):
            sum += val
            xval += val * index[0]
            yval += val * index[1]

        self.cofg = round(xval / sum), round(yval / sum)
        
        return self.cofg

    def swap_axes(self):
        """
        We adopt the convention that the axis of cylindrical symmetry is
        associated with the second index of the image array. This helper
        method swaps the axes of the image in case that is needed to comply
        with this convention.
        """
        numpy.transpose(self.image)

    def workspace_init(self, centre=None, *quadrants):
        """
        Set up the workspace for a subsequent inversion.

        centre is a tuple containing the pixel coordinates to be used as the
        image centre.

        We adopt the convention that the axis of cylindrical symmetry is
        associated with the second index of the image array. If necessary,
        call the method swap_axes before setting up a workspace to rotate the
        image. 

        The last argument(s) indicate the quadrants to be used for
        constructing the workspace. The quadrants are numbered 0-3:

        Quadrant 0: from centre to (xmax, ymax) [Top right]
        Quadrant 1: from centre to (xmax, 0)    [Bottom right]
        Quadrant 2: from centre to (0, 0)       [Bottom Left]
        Quadrant 3: from centre to (0, ymax]    [Top left]
        """
        self.centre = centre

        # Set up views into each quadrant with correct indexing away
        # from image centre. The way these are setup assumes that the
        # centre is right on the lower left corner ofthe centre pixel.
        self.quadrant = [
            self.image[centre[0]::, centre[1]::], 
            self.image[centre[0]::, centre[1]-1::-1],
            self.image[centre[0]-1::-1, centre[1]-1::-1],
            self.image[centre[0]-1::-1, centre[1]::]
            ]

        # Find largest dimension in x and y for the relevant quadrants
        xdim = max ([self.quadrant[i].shape[0] for i in quadrants])
        ydim = max ([self.quadrant[i].shape[1] for i in quadrants])

        self.workspace=numpy.zeros((xdim, ydim))
        
        # Average together the requested quadrants. Take care of
        # situation where a workspace pixel doesn't have a
        # corresponding pixel in all 4 quadrants.
        for (i, j), val in numpy.ndenumerate(self.workspace):
            norm = 0
            for q in quadrants:
                if (i < self.quadrant[q].shape[0] and 
                    j < self.quadrant[q].shape[1]):
                    self.workspace[i][j] += self.quadrant[q][i][j]
                    norm += 1
            self.workspace[i][j] /= norm

if __name__ == "__main__":
    import sys

    img=VMI()
    file=sys.argv[1]

    try:
        img.read(file)
    except IOError:
        print "Could not read file", file
        sys.exit(74)

    img.centre_of_gravity()
    print "Centre of gravity:", img.cofg 
    
    img.workspace_init(img.cofg, 0, 1, 2, 3)

    import pylab
    pylab.figure()
    pylab.imshow(img.workspace, origin='lower')
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


    
