/* TODO: 
   Use a better type than unsigned short for boolenas
   Use epsabs, rather than setting val=o in integrand for small values
*/

/* Note that Python.h must be included before any other header files. */
#include <Python.h>

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

  /* Round small values to 0 otherwise the integration error becomes dominated
     by the numerical error in exp such that the relative error is huge and the
     integration fails. */
  if (fabs(val) > __SMALL)
    return val;
  else
    return 0.0;
}

static PyObject *
matrix(PyObject *self, PyObject *args)
{
  int lmax, kmax, Rbins, Thetabins;
  double sigma, epsabs, epsrel; /* Suggest epsabs = 0.0, epsrel = 1.0e-7 */   
  double rkspacing, dTheta;
  int wkspsize; /* Suggest: wkspsize = 100000. */
  int ldim, kdim, midTheta, k;
  unsigned short int oddl, ThetabinsOdd, linc;
  npy_intp dims[4];
  PyArrayObject *matrix;
  gsl_integration_workspace *wksp;
  gsl_function fn;
  int_params params;
  gsl_integration_qaws_table *table;

  if (!PyArg_ParseTuple(args, "iiiidHddi",
			&kmax, &lmax, &Rbins, &Thetabins, &sigma, &oddl, &epsabs, &epsrel, &wkspsize))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to matrix");
      return NULL;
    }

  kdim = kmax + 1;

  /* If oddl is 0 (false), we only consider even harmonics, and adjust the
     indexing accordingly. */
  if (oddl)
    {
      ldim = lmax + 1;
      linc = 1;
    }
  else
    {
      if (GSL_IS_ODD(lmax))
	{
	  PyErr_SetString (PyExc_TypeError, "oddl is 0 (false), but lmax is odd.");
	  return NULL;
	}
      else /* lmax is even */
	{
	  ldim = (lmax / 2) + 1;
	  linc = 2;
	}
    }

  /* Allocate a new numpy array for the pbasex matrix */
  dims[0] = (npy_intp) kdim;
  dims[1] = (npy_intp) ldim;
  dims[2] = (npy_intp) Rbins;
  dims[3] = (npy_intp) Thetabins;

  matrix =(PyArrayObject *) PyArray_SimpleNew (4, dims, NPY_DOUBLE);
  if (!matrix)
    {
      Py_XDECREF (detectfn_coef);
      return PyErr_NoMemory();
    }
  
  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  wksp = gsl_integration_workspace_alloc(wkspsize);
  if (!wksp)
    {
      Py_XDECREF (detectfn_coef);
      Py_DECREF (matrix);
      return PyErr_NoMemory();
    }

  table = gsl_integration_qaws_table_alloc(-0.5, 0.0, 0.0, 0.0);
  if (!table)
    {
      Py_DECREF (matrix);
      gsl_integration_workspace_free(wksp);
      return PyErr_NoMemory();
    }

  fn.function = &integrand;
  fn.params = &params;

  rkspacing = ((double) Rbins) / kdim;

  params.two_sigma2 = 2.0 * sigma * sigma;

  /* We create a matrix for Theta = -Pi..Pi inclusive of both endpoints, despite
     the redundancy of the last (or first) endpoint. We use the symmetry of the
     Legendre polynomials P_L(cos(theta))=P_L(cos(-theta)) to calculate the
     points in the range 0..Pi from those in the range -Pi..0. 

     If Thetabins is an odd number, there will be a point at theta = 0, for
     which we don't need to use the symmetry condition. If Thetabins is an even
     number, then there won't be a point at Theta = 0. midTheta defines the last
     point for which we need to do the calculation in either case.
  */
  dTheta = 2.0 * M_PI / (Thetabins - 1);
  midTheta = Thetabins / 2; /* Intentionally round down. */
  
  if (GSL_IS_EVEN(Thetabins))
    {
      midTheta--;
      ThetabinsOdd = 0;
    }
  else
    {
      ThetabinsOdd = 1;
    }

  for (k=0; k<=kmax; k++)
    {
      int l;
      double upper_bound;
      
      params.rk = k * rkspacing;
      upper_bound = params.rk + __UPPER_BOUND_FACTOR * sigma;

      for (l = 0; l <= lmax; l += linc)
	{
	  int R;
	  params.l = l;
	  for (R = 0; R < Rbins; R++)
	    {
	      int j;
	      params.R = (double) R;
	      for (j = 0; j <= midTheta; j++)
		{
		  int status;
		  double result, abserr;
		  double Theta = -M_PI + j * dTheta;

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

		  switch (status)
		    {
		      PyObject *pval;
		      char *errstring;

		    case GSL_SUCCESS:
		      pval = PyFloat_FromDouble (result);

		      if (!pval)
			goto fail;

		      if (PyArray_SETITEM(matrix, PyArray_GETPTR4(matrix, k, l, R, j), pval)) 
			goto fail;

		      /* Symmetry of Legendre polynomials is such that
			 P_L(cos(Theta))=P_L(cos(-Theta)), so we can exploit
			 that here unless Theta = 0 (which only occurs if
			 Thetabins is odd), in which case it's not needed.
		      */
		      if (!(ThetabinsOdd && j == midTheta))
			if (PyArray_SETITEM(matrix, PyArray_GETPTR4(matrix, k, l, R, Thetabins - j - 1), pval)) 
			  goto fail;

		      Py_DECREF(pval);
		      break;

		    case GSL_EMAXITER:
		      if (!asprintf(&errstring,
				    "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %d Theta: %f\n", 
				    k, l, R, Theta))
			{
			  PyErr_SetString (IntegrationError, errstring);
			  free (errstring);
			}
		      else
			{
			  PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail\n");
			}

		      goto fail;
		      
		    case GSL_EROUND:
		      if(!asprintf(&errstring,
				   "Failed to integrate: round-off error.\nk: %d l: %d R: %d Theta: %f\n", 
				   k, l, R, Theta))
			{
			  PyErr_SetString (IntegrationError, errstring);
			  free (errstring);
			}
		      else
			{
			  PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail\n");
			}
		      goto fail;
		      
		    case GSL_ESING:
		      if (!asprintf(&errstring,
				    "Failed to integrate: singularity.\nk: %d l: %d R: %d Theta: %f\n", 
				    k, l, R, Theta))
			{
			  PyErr_SetString (IntegrationError, errstring);
			  free (errstring);
			}
		      else
			{
			  PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail\n");
			}
		      goto fail;
		      
		    case GSL_EDIVERGE:
		      if (!asprintf(&errstring,
				    "Failed to integrate: divergent.\nk: %d l: %d R: %d Theta: %f\n", 
				    k, l, R, Theta))
			{
			  PyErr_SetString (IntegrationError, errstring);
			  free (errstring);
			}
		      else
			{
			  PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail\n");
			}
		      goto fail;
		      
		    default:
		      if (!asprintf(&errstring,
				    "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %d Theta: %f\n", 
				    status, k, l, R, Theta))
			{
			  PyErr_SetString (IntegrationError, errstring);
			  free (errstring);
			}
		      else
			{
			  PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail\n");
			}
		      goto fail;
		    }	
		}

	    }
	}
    }

  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);

  return (PyObject *)matrix;

 fail:  
  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);
  Py_DECREF(matrix);

  return NULL;
}

static PyObject *
basisfn(PyObject *self, PyObject *args)
/* Calculates a single (k, l) PBASEX basis function and returns it as
   a two dimensional Numpy array Python object. */
{
  int Rbins, Thetabins, k, l, R;
  int wkspsize; /* Suggest: wkspsize = 100000. */
  int midTheta;
  double sigma, epsabs, epsrel; /* Suggest epsabs = 0.0, epsrel = 1.0e-7 */   
  double dTheta, rk, upper_bound;
  unsigned short int ThetabinsOdd;
  npy_intp dims[2];
  double *matrix = NULL;
  gsl_integration_workspace *wksp;
  gsl_function fn;
  int_params params;
  gsl_integration_qaws_table *table;
  char *errstring;

  if (!PyArg_ParseTuple(args, "iiiiddddi",
			&k, &l, &Rbins, &Thetabins, &sigma, &rk, &epsabs, &epsrel, &wkspsize))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to matrix");
      return NULL;
    }

  /* Release GIL, as we're not accessing any Python objects for the main calculation. */
  NPY_BEGIN_ALLOW_THREADS

  fn.params = &params;
  fn.function = &integrand;

  matrix = malloc(Rbins * Thetabins * sizeof (double));
  if (matrix == NULL)
    return PyErr_NoMemory(); 
  
  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  wksp = gsl_integration_workspace_alloc(wkspsize);
  if (!wksp)
    {
      return PyErr_NoMemory();
    }

  table = gsl_integration_qaws_table_alloc(-0.5, 0.0, 0.0, 0.0);
  if (!table)
    {
      gsl_integration_workspace_free(wksp);
      return PyErr_NoMemory();
    }


  params.two_sigma2 = 2.0 * sigma * sigma;
  params.rk = rk;
  params.l = l;

  /* We create a matrix for Theta = -Pi..Pi inclusive of both endpoints, despite
     the redundancy of the last (or first) endpoint. We use the symmetry of the
     Legendre polynomials P_L(cos(theta))=P_L(cos(-theta)) to calculate the
     points in the range 0..Pi from those in the range -Pi..0. 

     If Thetabins is an odd number, there will be a point at theta = 0, for
     which we don't need to use the symmetry condition. If Thetabins is an even
     number, then there won't be a point at Theta = 0. midTheta defines the last
     point for which we need to do the calculation in either case.
  */

  dTheta = 2.0 * M_PI / (Thetabins - 1);
  midTheta = Thetabins / 2; /* Intentionally round down. */
  
  if (GSL_IS_EVEN(Thetabins))
    {
      midTheta--;
      ThetabinsOdd = 0;
    }
  else
    {
      ThetabinsOdd = 1;
    }

  upper_bound = rk + __UPPER_BOUND_FACTOR * sigma;
  
  for (R = 0; R < Rbins; R++)
    {
      int j;
      params.R = (double) R;

      for (j = 0; j <= midTheta; j++)
	{
	  int status;
	  double result, abserr;
	  double Theta = -M_PI + j * dTheta;

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
	  
	  switch (status)
	    {
	      int ret;

	    case GSL_SUCCESS:
	      matrix[R * Rbins + j] = result;

	      /* Symmetry of Legendre polynomials is such that
		 P_L(cos(Theta))=P_L(cos(-Theta)), so we can exploit
		 that here unless Theta = 0 (which only occurs if
		 Thetabins is odd), in which case it's not needed.
	      */
	      if (!(ThetabinsOdd && j == midTheta))
		matrix[R * Rbins + Thetabins - j - 1] = result;

	      break;
	      
	    case GSL_EMAXITER:
	      ret = asprintf(&errstring,
			     "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %d Theta: %f\n", 
			     k, l, R, Theta);
	      if (ret < 0)
		errstring = NULL;
	      
	      goto integration_fail;
	      
	    case GSL_EROUND:
	      ret = asprintf(&errstring,
			     "Failed to integrate: round-off error.\nk: %d l: %d R: %d Theta: %f\n", 
			     k, l, R, Theta);
	      if (ret < 0)
		errstring = NULL;
	      
	      goto integration_fail;
	      
	    case GSL_ESING:
	      ret = asprintf(&errstring,
			     "Failed to integrate: singularity.\nk: %d l: %d R: %d Theta: %f\n", 
			     k, l, R, Theta);
	      if (ret < 0)
		errstring = NULL;

	      goto integration_fail;
	      
	    case GSL_EDIVERGE:
	      ret = asprintf(&errstring,
			     "Failed to integrate: divergent.\nk: %d l: %d R: %d Theta: %f\n", 
			     k, l, R, Theta);
	      if (ret < 0)
		errstring = NULL;

	      goto integration_fail;
	      
	    default:
	      ret = asprintf(&errstring,
			     "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %d Theta: %f\n", 
			     status, k, l, R, Theta);
	      if (ret < 0)
		errstring = NULL;

	      goto integration_fail;
	    }	
	}
    }

  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);

  /* Regain GIL as we'll now access some Python objects. */
  NPY_END_ALLOW_THREADS

  /* Make a numpy array from matrix. */
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

 integration_fail:  
  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);
  free (matrix);

  if (errstring != NULL)
    {
      PyErr_SetString (IntegrationError, errstring);
      free (errstring);
    }
  else
    {
      PyErr_SetString (IntegrationError, "Couldn't allocate storage for further detail\n");
    }

  return NULL;
}

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef MatrixMethods[] = {
    {"basisfn",  basisfn, METH_VARARGS,
     "Returns a matrix of a single (k, l) basis function."},
    {"matrix",  matrix, METH_VARARGS,
     "Returns an inversion matrix of basis functions."},
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
