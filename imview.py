#!/usr/bin/python

import numpy
import pylab
import matplotlib

data=numpy.loadtxt("dataset.dat")
pylab.figure()
pylab.imshow(data, cmap=pylab.cm.gist_gray)
pylab.show()
