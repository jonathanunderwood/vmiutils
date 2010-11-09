/* TODO: 
   Use a better type than unsigned short for boolenas
   Use epsabs, rather than setting val=o in integrand for small values
*/

/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <numpy/ndarrayobject.h>
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
basisfn(PyObject *self, PyObject *args)
{
  int l;
  double r, rk, sigma, theta, rad, ang, a, s;

  if (!PyArg_ParseTuple(args, "dddid", &r, &rk, &sigma, &l, &theta))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to basisfn_full");
      return NULL;
    }

  a = r - rk;
  s = 2.0 * sigma * sigma;
  rad = exp(-(a * a) / s);
  ang = gsl_sf_legendre_Pl(l, cos(theta));

  return Py_BuildValue("d", rad * ang);
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
  PyObject *matrix;
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

  /* matrix = malloc (Rbins * Thetabins * kdim * ldim * sizeof(double)); */
  /* if (!matrix) */
  /*   return PyErr_NoMemory(); */
  /* create numpy array to hold the matrix. */
  dims[0] = (npy_intp) kdim;
  dims[1] = (npy_intp) ldim;
  dims[2] = (npy_intp) Rbins;
  dims[3] = (npy_intp) Thetabins;

  matrix = PyArray_SimpleNew (4, dims, NPY_DOUBLE);
  if (!matrix)
    return PyErr_NoMemory();
  
  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  wksp = gsl_integration_workspace_alloc(wkspsize);
  if (!wksp)
    {
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
		      int ret;

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
		      ret = asprintf(&errstring,
			       "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta);
		      PyErr_SetString (IntegrationError, errstring);
		      free (errstring);
		      goto fail;
		      
		    case GSL_EROUND:
		      ret = asprintf(&errstring,
			       "Failed to integrate: round-off error.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta);
		      PyErr_SetString (IntegrationError, errstring);
		      free (errstring);
		      goto fail;
		      
		    case GSL_ESING:
		      ret = asprintf(&errstring,
			       "Failed to integrate: singularity.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta);
		      PyErr_SetString (IntegrationError, errstring);
		      free (errstring);
		      goto fail;
		      
		    case GSL_EDIVERGE:
		      ret = asprintf(&errstring,
			       "Failed to integrate: divergent.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta);
		      PyErr_SetString (IntegrationError, errstring);
		      goto fail;
		      
		    default:
		      ret = asprintf(&errstring,
			       "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %d Theta: %f\n", 
			       status, k, l, R, Theta);
		      PyErr_SetString (IntegrationError, errstring);
		      free (errstring);
		      goto fail;
		    }	
		}

	    }
	}
    }

  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);

  return matrix;

 fail:  
  gsl_integration_workspace_free(wksp);
  gsl_integration_qaws_table_free(table);
  Py_DECREF(matrix);

  return NULL;
}

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef MatrixMethods[] = {
    {"basisfn",  basisfn, METH_VARARGS,
     "Returns the value of a basis function."},
    {"matrix",  matrix, METH_VARARGS,
     "Returns an inversion matrix of basis functions."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _basisfn. */
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
