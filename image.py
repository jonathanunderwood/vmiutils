import numpy

def centre_of_gravity(image):
    """
    Returns the pixel corresponding to the centre of gravity of a VMI image
    stored as a numpy array. 
    """
    sum = 0.0
    xval = 0.0
    yval = 0.0

    for index, val in numpy.ndenumerate(image):
        sum += val
        xval += val * index[0]
        yval += val * index[1]

    cofg = int(round(xval / sum)), int(round(yval / sum))
        
    return cofg

def workspace_init(image, centre, *quadrants):
    """
    Set up a workspace for a subsequent inversion.
    
    centre is a tuple containing the pixel coordinates to be used as the image
    centre.
    
    We adopt the convention that the axis of cylindrical symmetry is
    associated with the second index of the numpy image array.

    The last argument(s) indicate the quadrants to be used for constructing
    the workspace. The quadrants are numbered 0-3:
    
    Quadrant 0: from centre to (xmax, ymax) [Top right]
    Quadrant 1: from centre to (xmax, 0)    [Bottom right]
    Quadrant 2: from centre to (0, 0)       [Bottom Left]
    Quadrant 3: from centre to (0, ymax]    [Top left]
    """
    # Set up views into each quadrant with correct indexing away from image
    # centre. The way these are setup assumes that the centre is right on the
    # lower left corner ofthe centre pixel.
    quadrant = [
        image[centre[0]::, centre[1]::], 
        image[centre[0]::, centre[1]-1::-1],
        image[centre[0]-1::-1, centre[1]-1::-1],
        image[centre[0]-1::-1, centre[1]::]
        ]

    # Find largest dimension in x and y for the relevant quadrants
    dims = (max([quadrant[i].shape[0] for i in quadrants]), 
            max([quadrant[i].shape[1] for i in quadrants]))

    workspace=numpy.zeros(dims)
        
    # Average together the requested quadrants. Take care of situation where a
    # workspace pixel doesn't have a corresponding pixel in all the quadrants.
    for (i, j), val in numpy.ndenumerate(workspace):
        norm = 0
        for q in quadrants:
            if ((i < quadrant[q].shape[0]) and 
                (j < quadrant[q].shape[1])):
                workspace[i][j] += quadrant[q][i][j]
                norm += 1
        workspace[i][j] /= norm

    return workspace

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


    
