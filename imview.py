#!/usr/bin/python

import numpy
import pylab
import matplotlib

data=numpy.loadtxt("im_example.asc")
pylab.figure()
pylab.imshow(data, cmap=pylab.cm.gist_gray)
pylab.show()
