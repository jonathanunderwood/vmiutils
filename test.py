import sys
import numpy
import pylab
import vmi

file = sys.argv[1]

try:
    image = numpy.loadtxt(file)
except IOError:
    print "Could not read file", file
    sys.exit(74)

cimage = vmi.VMICartesianImage()
cimage.from_numpy_array(image)
print 'centre of gravity:', cimage.centre_of_gravity()
print 'centre of grid:', cimage.centre_of_grid()
cimage.set_centre(cimage.centre_of_grid())

pylab.figure()
pylab.imshow(cimage.image, origin='lower')
pylab.show()

pimage = vmi.VMIPolarImage()
pimage.from_VMICartesianImage(cimage)

cimage2 = vmi.VMICartesianImage()
cimage2.from_VMIPolarImage(pimage)

pylab.figure()
pylab.imshow(cimage2.image, origin='lower')
pylab.show()
