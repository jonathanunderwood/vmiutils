#!/usr/bin/python

import numpy
import pylab
import matplotlib
import vmiutils as v

data=numpy.loadtxt("im_example.asc")
pylab.figure()
pylab.imshow(data, cmap=pylab.cm.gist_gray)
pylab.show()

datac = v.CartesianImage()
datac.from_numpy_array(data)
datac.set_centre(datac.centre_of_gravity())

datap = v.PolarImage()
datap.from_CartesianImage(datac)

#pylab.figure()
#pylab.axes(polar=True)
#pylab.pcolormesh(datap.R, datap.Theta, datap.image)
#pylab.show()

pylab.figure()
pylab.imshow(datap.image, cmap=pylab.cm.gist_gray)
pylab.show()

