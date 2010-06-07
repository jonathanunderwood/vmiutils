from distutils.core import setup, Extension

setup(name='VMIinvert',
      version='0.1'
      description='Programs for inverting VMI image',
      author='Jonathan G. Underwood',
      author_email='j.underwood@ucl.ac.uk',
      packages=['pbasex', 'ChoNa'],
      ext_modules=[Extension('pbasex.basisfn', 
                             ['pbasex/basisfn.c'],
                             libraries=['gsl', 'gslcblas', 'm'])],
      )
