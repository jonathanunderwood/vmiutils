from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext

# http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(name='VMIutils',
      version='0.2.0',
      description='Programs for inverting VMI image',
      author='Jonathan G. Underwood',
      author_email='j.underwood@ucl.ac.uk',
      cmdclass={'build_ext':build_ext},
      setup_requires=['numpy',],
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'futures',],
      packages=['vmiutils',
                'vmiutils.pbasex',
                'vmiutils.pbasex.detfn1',
                'vmiutils.ChoNa',
                'vmiutils.simulate',
            ],
      ext_modules=[Extension('vmiutils.pbasex._matrix', 
                             ['vmiutils/pbasex/_matrix.c'],
                             libraries=['gsl', 'gslcblas', 'm'],),
                   Extension('vmiutils.pbasex.detfn1._matrix_detfn1',
                             ['vmiutils/pbasex/detfn1/_matrix_detfn1.c'],
                             libraries=['gsl', 'gslcblas', 'm'],),
                   Extension('vmiutils.pbasex._fit',
                             ['vmiutils/pbasex/_fit.c'],
                             libraries=['gsl', 'gslcblas', 'm'],),
                             ],
      scripts=['scripts/pbmatrix',
               'scripts/pbmatrix_detfn1',
               'scripts/pbfit',
               'scripts/pbfitplt',
               'scripts/pbfittocsv',
               'scripts/cnfit',
               'scripts/vmiview',
               'scripts/vmicentrefix',
           ],
      )
