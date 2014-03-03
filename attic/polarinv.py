import numpy
import scipy.interpolate as spint
import math





if __name__ == "__main__":
    import sys
    import pylab

    file=sys.argv[1]

    try:
        image = numpy.loadtxt(file)
    except IOError:
        print "Could not read file", file
        sys.exit(74)

    vmi = VMImage(image)
    cofg = vmi.centre_of_gravity()
    print "Centre of gravity:", cofg[0], cofg[1] 

    vmi.polar_workspace_init(centre=cofg)

    pylab.figure()
    pylab.imshow(vmi.image, origin='lower')
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
