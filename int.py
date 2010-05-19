# Simple file containing stuff for timing scipy.integrate and seeing if we can
# speed things up with cython.

import scipy.integrate as sint
import cProfile as prof
import math as m

def func(x):
    return x * m.sin(x)

p=prof.Profile()

prof.run("sint.quad(func, 0.0, 300.0)")
#p.sort_stats('cumulative')
#p.print_stats(10)


