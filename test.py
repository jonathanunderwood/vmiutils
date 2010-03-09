import sys
import vmipolar
import vmicart
import numpy
import pylab

file = sys.argv[1]

try:
    image = numpy.loadtxt(file)
except IOError:
    print "Could not read file", file
    sys.exit(74)

cimage = vmicart.VMICartesianImage()
cimage.from_numpy_array(image)
print 'centre of gravity:', cimage.centre_of_gravity()

pylab.figure()
pylab.imshow(cimage.image, origin='lower')
pylab.show()

pimage = vmipolar.VMIPolarImage()
pimage.from_VMICartesianImage(cimage)

cimage2 = vmicart.VMICartesianImage()
cimage2.from_VMIPolarImage(pimage)

pylab.figure()
pylab.imshow(cimage2.image, origin='lower')
pylab.show()
