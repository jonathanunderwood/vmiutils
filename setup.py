from distutils.core import setup, Extension

setup(name='VMIutils',
      version='0.1',
      description='Programs for inverting VMI image',
      author='Jonathan G. Underwood',
      author_email='j.underwood@ucl.ac.uk',
      packages=['vmiutils', 'vmiutils.pbasex', 'vmiutils.ChoNa'],
      ext_modules=[Extension('vmiutils.pbasex._basisfn', 
                             ['vmiutils/pbasex/_basisfn.c'],
                             libraries=['gsl', 'gslcblas', 'm'])],
      scripts=['scripts/pbmatrix', 'scripts/pbfit'],
      )
