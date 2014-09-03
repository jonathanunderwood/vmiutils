/* Copyright (C) 2014 by Jonathan G. Underwood.
 *
 * This file is part of VMIUtils.
 *
 * VMIUtils is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * VMIUtils is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with VMIUtils.  If not, see <http://www.gnu.org/licenses/>.
 */

/* TODO: 
   Use a better type than unsigned short for boolenas
   Use epsabs, rather than setting val=o in integrand for small values
*/

/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <stdio.h>
#include <limits.h>

/* For numpy we need to specify the API version we're targetting so
   that deprecated API warnings are issued when appropriate. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#undef NPY_NO_DEPRECATED_API

#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_errno.h>

#define __SMALL 1.0e-30
#define __UPPER_BOUND_FACTOR 15.0

/* Exceptions for this module. */
static PyObject *IntegrationError;

/****************************************************************
 * Structures and functions for straightforward pBasex
 ****************************************************************/
typedef struct 
{
  int l;
  double R, RcosTheta, rk, two_sigma2;
} int_params;

static double integrand(double r, void *params)
/* Integrand for PBASEX, no detection function. */
{
  double a, rad, ang, val;
  int_params p = *(int_params *) params;

  a = r - p.rk;
  rad = exp (-(a * a) / p.two_sigma2);
  ang = gsl_sf_legendre_Pl (p.l, p.RcosTheta / r);

  val = r * rad * ang / sqrt(r + p.R);

  return val;

  /* Round small values to 0 otherwise the integration error becomes
     dominated by the numerical error in exp such that the relative
     error is huge and the integration fails. */
  /* if (fabs(val) > __SMALL) */
  /*   return val; */
  /* else */
  /*   return 0.0; */
}

static PyObject *
basisfn(PyObject *self, PyObject *args)
/* Calculates a single (k, l) PBASEX basis function and returns it as
   a two dimensional Numpy array Python object. */
{
  int Rbins, Thetabins, k, l, i;
  int wkspsize; /* Suggest: wkspsize = 100000. */
  int midThetahigh, midThetalow, jmax;
  double sigma, epsabs, epsrel; /* Suggest epsabs = 0.0, epsrel = 1.0e-7 */   
  double dTheta, rk, upper_bound;
  unsigned short int ThetabinsEven;
  npy_intp dims[2];
  double *matrix = NULL;
  gsl_integration_workspace *wksp;
  gsl_function fn;
  int_params params;
  gsl_integration_qaws_table *table;
  PyObject *numpy_matrix;

  if (!PyArg_ParseTuple(args, "iiiiddddi",
			&k, &l, &Rbins, &Thetabins, &sigma, &rk, &epsabs, &epsrel, &wkspsize))
    {
      PyErr_SetString (PyExc_TypeError, "Bad arguments to basisfn");
      return NULL;
    }

  /* Release GIL, as we're not accessing any Python objects for the
     main calculation. Any return statements will need to be preceeded
     by Py_BLOCK_THREADS - see /usr/include/pythonX/ceval.h */
  Py_BEGIN_ALLOW_THREADS

  fn.function = &integrand;
  fn.params = &params;

  matrix = malloc(Rbins * Thetabins * sizeof (double));
  if (matrix == NULL)
    {
      Py_BLOCK_THREADS
      return PyErr_NoMemory(); 
    }

  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  wksp = gsl_integration_workspace_alloc(wkspsize);
  if (!wksp)
    {
      free (matrix);
      Py_BLOCK_THREADS
      return PyErr_NoMemory();
    }

  table = gsl_integration_qaws_table_alloc(-0.5, 0.0, 0.0, 0.0);
  if (!table)
    {
      free (matrix);
      gsl_integration_workspace_free(wksp);
      Py_BLOCK_THREADS
      return PyErr_NoMemory();
    }

  params.two_sigma2 = 2.0 * sigma * sigma;
  params.rk = rk;
  params.l = l;

  /* We create a matrix for Theta = -Pi..Pi inclusive of both
     endpoints, despite the redundancy of the last (or first)
     endpoint.

     In what follows we create a matrix for Theta which lies in the
     range -Pi..Pi. The first bin starts with Theta=-Pi, and the last
     bin ends at Pi. In other words, we don't create an extra
     redundant bin at the end beginning at Pi. We also calulate the
     value for the bin by taking the value of Theta in the centre of
     the bin.

     We use the symmetry of the Legendre polynomials
     P_L(cos(theta))=P_L(cos(-theta)) to calculate the points in the
     range 0..Pi from those in the range -Pi..0.

     If Thetabins is an even number points are distributed such that
     we can also make use of the symmetry
     P_L(cos(theta))=(-1)^LP_L(cos(theta+pi), and as such we can halve
     the number of integrations needed.

     The following tables help with visualizing what's going on below.

     thetabins = 4
     thetabw = pi/2
     | bin val | -pi    | -pi/2 |    0 | pi/2  |
     | centres | -3pi/4 | -pi/4 | pi/4 | 3pi/4 |
     | idx     | 0      | 1     |    2 | 3     |

     thetabins = 6
     thetabw = pi/3
     | binval  | -pi    | -2pi/3 | -pi/3 |    0 | pi/3 | 2pi/3 |
     | centres | -5pi/6 | -pi/2  | -pi/6 | pi/6 | pi/2 | 5pi/6 |
     | idx     | 0      | 1      | 2     |    3 | 4    | 5     |

     thetabins = 8
     thetabw=pi/4
     | binval  | -pi    | -3pi/4 | -pi/2  | -pi/4 |    0 | pi/4  | pi/2  | 3pi/4 |
     | centres | -7pi/8 | -5pi/8 | -3pi/8 | -pi/8 | pi/8 | 3pi/8 | 5pi/8 | 7pi/8 |
     | idx     | 0      | 1      | 2      | 3     |    4 | 5     | 6     | 7     |

     thetabins = 3
     thetabw = 2pi/3
     | bin val | -pi    | -pi/3 | pi/3  |
     | centres | -2pi/3 |     0 | 2pi/3 |
     | idx     | 0      |     1 | 2     |

     thetabins = 5
     thetabw = 2pi/5
     | bin val | -pi    | -3pi/5 | -pi/5 | pi/5  | 3pi/5 |
     | centres | -4pi/5 | -2pi/5 |     0 | 2pi/5 | 4pi/5 |
     | idx     | 0      | 1      |     2 | 3     | 4     |

  */

  dTheta = 2.0 * M_PI / Thetabins;

  if (GSL_IS_EVEN(Thetabins))
    {
      ThetabinsEven = 1;
      jmax = (Thetabins - 1) / 4; /* Intentionally round down. */
      midThetalow = (Thetabins - 1) / 2; /* Intentionally round down. */
      midThetahigh = midThetalow + 1;
    }
  else
    {
      ThetabinsEven = 0;
      jmax = Thetabins / 2; /* Intentionally round down. */
      /* These are unused for the Thetabins odd case, but initialize
	 anyway to silence compiler. */
      midThetalow = INT_MIN;
      midThetahigh = INT_MIN;
    }

  upper_bound = rk + __UPPER_BOUND_FACTOR * sigma;
  
  for (i = 0; i < Rbins; i++)
    {
      int j;
      int dim1 = i * Thetabins;
      double R = i + 0.5;

      params.R = R;

      for (j = 0; j <= jmax; j++)
	{
	  int status;
	  double result, abserr;
	  double Theta = -M_PI + (j + 0.5) * dTheta;
	  
	  params.RcosTheta = R * cos (Theta);
	  
	  if (upper_bound > R)
	    {
	      status = gsl_integration_qaws (&fn, (double) R, upper_bound, table,
					     epsabs, epsrel, wkspsize,
					     wksp, &result, &abserr);
	    }
	  else 
	    {
	      status = GSL_SUCCESS;
	      result = 0.0;
	      abserr = 0.0;
	    }
	  
	  if (status == GSL_SUCCESS)
	    {
	      matrix[dim1 + j] = result;

	      /* Symmetry of Legendre polynomials is such that
		 P_L(cos(Theta))=P_L(cos(-Theta)). */
	      matrix[dim1 + Thetabins - j - 1] = result;

	      if (ThetabinsEven)
		{
		  double valneg;

		  if (l % 2) /* l is odd */
		    valneg = -result;
		  else /* l is even */
		    valneg = result;

		  matrix[dim1 + midThetahigh + j] = valneg;
		  matrix[dim1 + midThetalow - j] = valneg;
		}
	    }
	  else
	    {
	      char *errstring;

	      gsl_integration_workspace_free(wksp);
	      gsl_integration_qaws_table_free(table);
	      free (matrix);

	      switch (status)
		{
		case GSL_EMAXITER:
		  if (asprintf(&errstring,
			       "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %f Theta: %f\n",
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EROUND:
		  if (asprintf(&errstring,
			       "Failed to integrate: round-off error.\nk: %d l: %d R: %f Theta: %f\n",
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		case GSL_ESING:
		  if (asprintf(&errstring,
			       "Failed to integrate: singularity.\nk: %d l: %d R: %f Theta: %f\n",
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EDIVERGE:
		  if (asprintf(&errstring,
			       "Failed to integrate: divergent.\nk: %d l: %d R: %f Theta: %f\n",
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		default:
		  if (asprintf(&errstring,
			       "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %f Theta: %f\n",
			       status, k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		}
	      
	      Py_BLOCK_THREADS
	      if (errstring != NULL)
		{
		  PyErr_SetString (IntegrationError, errstring);
		  free (errstring);
		}
	      else
		{
		  PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail on this error, sorry\n");
		}
	      
	      return NULL;
	    }
	}
    }

  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);

  /* Regain GIL as we'll now access some Python objects. */
  NPY_END_ALLOW_THREADS

  /* Make a 2D numpy array from matrix such that it is indexed as matrix[Rbin, Thetabin]. */
  dims[0] = (npy_intp) Rbins;
  dims[1] = (npy_intp) Thetabins;

  numpy_matrix = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void *) matrix);

  if (!numpy_matrix)
    {
      /* PyArray_SimpleNewFromData should already raise an exception in
	 this event, so we just need to return NULL. */
      free (matrix);
      Py_XDECREF(numpy_matrix); /* Belt & braces */
      return NULL;
    }

  /* Ensure the correct memory deallocator is called when Python
     destroys numpy_matrix. In principle, it should be sufficient to
     just set the OWNDATA flag on numpy_matrix, but this makes the
     assumption that numpy uses malloc and free for memory management,
     which it may not in the future. Note also that we are only able
     to use malloc and free because these are guaranteed to give
     aligned memory for standard types (which double is). For more
     complicated objects we'd need to ensure we have aligned alloc and
     free. See: 
     http://blog.enthought.com/python/numpy/simplified-creation-of-numpy-arrays-from-pre-allocated-memory/
 */
  PyArray_SetBaseObject((PyArrayObject *) numpy_matrix, PyCObject_FromVoidPtr(matrix, free));

  return numpy_matrix;
}

/****************************************************************
 * Python extension module initialization stuff.
 ****************************************************************/

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef MatrixMethods[] = {
    {"basisfn",  basisfn, METH_VARARGS,
     "Returns a matrix of a single (k, l) pBasex basis function."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _matrix. */
PyMODINIT_FUNC
init_matrix(void)
{
  PyObject *mod;

  /* This is needed for the numpy API. */
  import_array();

  mod = Py_InitModule("_matrix", MatrixMethods);
  if (mod == NULL)
    return;

  /* Exceptions. */
  IntegrationError = PyErr_NewException("_matrix.IntegrationError", NULL, NULL);
  Py_INCREF(IntegrationError);
  PyModule_AddObject(mod, "IntegrationError", IntegrationError);
}

#undef __SMALL
#undef __UPPER_BOUND_FACTOR 
