#!/usr/bin/env python

# Deal with the non conformant 16 bit greyscale tif files from the Imperial
# camera.

import os
import sys
import numpy 
import Image

for file in sys.argv[1:]:
    # See http://www.mail-archive.com/numpy-discussion@scipy.org/msg07554.html
    print file
    image = Image.open(file)
    a = numpy.array(image.getdata())

    # For some reason row/column order is wrong, so need this:
    a = a.reshape((image.size[1],image.size[0]))

    root, ext = os.path.splitext(file)
    outfile = root + ".dat"
    
    numpy.savetxt(outfile, a, "%d")


# To display
#import pylab
#pylab.figure()
#pylab.imshow(a)
#pylab.show()
