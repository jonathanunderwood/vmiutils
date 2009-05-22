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

        for index, val in numpy.ndenumerate(self.data):
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

    def workspace_init(self, centre=None, *quadrants):
        """
        Set up the workspace for a subsequent inversion.

        centre is a tuple containing the pixel coordinates to be used as the
        image centre.

        We adopt the convention that the axis of cylindrical symmetry is
        associated with the second index of the image array. If necessary,
        call the method swap_axes before setting up a workspace to rotate the
        image. 

        This also sets up the four quadrant views into the image array.
        """
        self.centre = centre

        # Ensure the symmetry axis is along the y-axis of the image
        if (sym_axis == "x"):
            numpy.transpose(self.image)
        
        # Set up views into each quadrant with correct indexing away from
        # image centre. The way these are setup actually assumes that the
        # centre is right on the lower left corner ofthe centre pixel.
        self.quadrant = [self.image[centre[0]::, centre[1]::), 
                         self.image[centre[0]::, centre[1]-1::-1),
                         self.image[centre[0]-1::-1, centre[1]::),
                         self.image[centre[0]-1::-1, centre[1]-1::-1)]

        # Find largest dimension in x and y for the four quadrants
        maxdim_x = max ([self.quadrant[i].shape[0] for i in range(4)])
        maxdim_y = max ([self.quadrant[i].shape[1] for i in range(4)])

        # Find smallest dimension in x and y for the four quadrants
        mindim_x = min ([self.quadrant[i].shape[0] for i in range(4)])
        mindim_y = min ([self.quadrant[i].shape[1] for i in range(4)])

        

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
    print "Centre of gravity pixel:", img.cofg_pixel 
    



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


    
