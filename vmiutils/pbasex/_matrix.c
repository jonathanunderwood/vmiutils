/* TODO: 
   Use a better type than unsigned short for boolenas
   Use epsabs, rather than setting val=o in integrand for small values
*/

/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <stdio.h>

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

  /* Round small values to 0 otherwise the integration error becomes
     dominated by the numerical error in exp such that the relative
     error is huge and the integration fails. */
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
			&kmax, &lmax, &Rbins, &Thetabins, &sigma, &oddl, 
			&epsabs, &epsrel, &wkspsize))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to matrix");
      return NULL;
    }

  kdim = kmax + 1;

  /* If oddl is 0 (false), we only consider even harmonics, and adjust
     the indexing accordingly. */
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
      return PyErr_NoMemory();
    }
  
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
	  
	  if (status == GSL_SUCCESS)
	    {
	      matrix[R * Thetabins + j] = result;

	      /* Symmetry of Legendre polynomials is such that
		 P_L(cos(Theta))=P_L(cos(-Theta)), so we can exploit
		 that here unless Theta = 0 (which only occurs if
		 Thetabins is odd), in which case it's not needed.
	      */
	      if (!(ThetabinsOdd && j == midTheta))
		matrix[R * Thetabins + Thetabins - j - 1] = result;
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
			       "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EROUND:
		  if (asprintf(&errstring,
			       "Failed to integrate: round-off error.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		case GSL_ESING:
		  if (asprintf(&errstring,
			       "Failed to integrate: singularity.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EDIVERGE:
		  if (asprintf(&errstring,
			       "Failed to integrate: divergent.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		default:
		  if (asprintf(&errstring,
			       "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %d Theta: %f\n", 
			       status, k, l, R, Theta))
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
 * Structures and functions for pBasex incorporating a detection
 * function specified as a previous pBasex fit.
 ****************************************************************/
typedef struct 
{
  int l, df_linc;
  long df_kmax, df_lmax;
  double R, rk, two_sigma2, RcosTheta, RsinTheta;
  double df_sigma, df_two_sigma2, df_rkstep, df_rmax;
  double df_alpha, df_cos_beta, df_sin_beta;
  double *df_coefs, *df_legpol;
} int_params_detfn1;


static double integrand_detfn1 (double r, void *params)
/* Integrand for PBASEX with a detection function defined by a
   previous PBASEX fit. Here we calculate the detection frame theta
   angle using Zare Eq 3.86 from theta and phi in the lab frame, and
   the angles alpha and beta between the lab frame and detection
   frame.*/
{
  int_params_detfn1 p = *(int_params_detfn1 *) params;
  double theta = p.RcosTheta / r;
  double cos_theta = cos (theta);
  double sin_theta = sin (theta);
  double phi = asin(p.RsinTheta / (r * sin_theta));
  double cos_theta_det_frame = p.df_cos_beta * cos_theta + 
    p.df_sin_beta * sin_theta * cos (phi - p.df_alpha);
  double a, rad, ang, val, df_val, delta;
  int k, l, df_lidx, kmin, kmax;

  /* Usual basis function for pbasex */
  a = r - p.rk;
  rad = exp (-(a * a) / p.two_sigma2);
  ang = gsl_sf_legendre_Pl (p.l, cos_theta);

  /* Detection function at this r, theta.*/
  df_lidx = -1;
  for (l = 0; l <= p.df_lmax; l += p.df_linc)
    {
      df_lidx++;
      p.df_legpol[df_lidx] = gsl_sf_legendre_Pl(l, cos_theta_det_frame);
    }

  /* TODO: we could save time here by only considering radial basis
     functions within say 5 sigma of this value of r */
  delta = 5.0 * p.df_sigma;
  kmin = floor((r - delta) / p.df_rkstep);
  kmax = ceil((r + delta) / p.df_rkstep);
  if (kmin < 0)
    kmin = 0;
  if (kmax > p.df_kmax)
    kmax = p.df_kmax;

  df_val = 0.0;
  for (k = kmin; k <= kmax; k++)
    {	      
      double rk = k * p.df_rkstep;
      double a = r - rk;
      double rad = exp(-(a * a) / p.df_two_sigma2);

      df_lidx = -1;

      for (l = 0; l <= p.df_lmax; l += p.df_linc)
	{
	  double c1 = p.df_coefs[k * (p.df_lmax + 1) + l];
	  double c2;

	  df_lidx ++;
	  c2 = p.df_legpol[df_lidx];
	  df_val += rad * c1 * c2;
	}
    }

  val = df_val * rad * ang * r / sqrt(r + p.R);

  /* Round small values to 0 otherwise the integration error becomes dominated
     by the numerical error in exp such that the relative error is huge and the
     integration fails. */
  if (fabs(val) > __SMALL)
    return val;
  else
    return 0.0;
}

static int
get_pyfloat_attr (PyObject *obj, const char *attr, double *val)
/* Retrieve a float attribute attr from object obj and store it as a
   double precision number at val. Returns 0 on success, -1 otherwise. */
{
  PyObject *p = PyObject_GetAttrString(obj, attr);

  if (p == NULL)
    {
      Py_XDECREF(p); /* Belt & braces */
      return -1;
    }
  
  *val = PyFloat_AsDouble (p);
  Py_DECREF(p);

  if (PyErr_Occurred())
    return -1;

  return 0;
}

static int
get_pyint_attr (PyObject *obj, const char *attr, long *val)
/* Retrieve an python integer attribute attr from object obj and store
   it as a long integer number at val. Returns 0 on success, -1
   otherwise. */
{
  PyObject *p = PyObject_GetAttrString(obj, attr);

  if (p == NULL)
    {
      Py_XDECREF(p); /* Belt & braces */
      return -1;
    }
  
  /* TODO: for Python 3 we'll need to use PyLong_AsLong here. */
  *val = PyInt_AsLong (p);
  Py_DECREF(p);

  if (PyErr_Occurred())
    return -1;

  return 0;
}

static int
get_pybool_attr (PyObject *obj, const char *attr, int *val)
/* Retrieve a python boolean attribute attr from object obj. Stores 1
   at val if the boolean is True, 0 otherwise. Returns 0 on success,
   -1 otherwise (i.e. if error occurs). */
{
  PyObject *p = PyObject_GetAttrString(obj, attr);
  int bool;

  if (p == NULL)
    {
      Py_XDECREF(p); /* Belt & braces */
      return -1;
    }
  
  bool = PyObject_IsTrue (p);

  Py_DECREF(p);

  if (PyErr_Occurred())
    return -1;
  
  *val = bool;
  
  return 0;
}

static PyObject *
basisfn_detfn1(PyObject *self, PyObject *args)
/* Calculates a single (k, l) PBASEX basis function and returns it as
   a two dimensional Numpy array Python object. This includes a
   detection function specified as a previous PBASEX fit. */
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
  int_params_detfn1 params;
  gsl_integration_qaws_table *table;
  PyObject *numpy_matrix, *detectfn;
  size_t df_ldim;
  double alpha, beta;
  PyObject *pbasex_fit_module, *pbasex_fit_module_dict, *pbasex_fit_class;
  unsigned short int correct_type;
  int df_oddl;

  if (!PyArg_ParseTuple(args, "iiiiddddiOdd",
			&k, &l, &Rbins, &Thetabins, &sigma, &rk, &epsabs, &epsrel, &wkspsize,
			&detectfn, &alpha, &beta))
    {
      PyErr_SetString (PyExc_TypeError, "Bad arguments to basisfn");
      return NULL;
    }

  /* detectionfn is a python object containing a representation of the
     detection function as a pbasex fit, and so assumes the detection
     function has cylindrical symmetry about some axis. (alpha, beta)
     specify the angles between the lab frame symmetry axis, and the
     detection frame symmetry axis. */
  pbasex_fit_module = PyImport_ImportModule("vmiutils.pbasex.fit"); /* New reference */
  pbasex_fit_module_dict = PyModule_GetDict(pbasex_fit_module); /* Borrowed reference */
  pbasex_fit_class = PyDict_GetItemString(pbasex_fit_module_dict, "PbasexFit"); /* Borrowed reference */
  correct_type = PyObject_IsInstance (detectfn, pbasex_fit_class);

  Py_XDECREF(pbasex_fit_module);

  if (!correct_type)
    {
      PyErr_SetString (PyExc_TypeError, "Unrecognized object for detection function");
      return NULL;
    }
  else
    {
      PyObject *p = NULL;
      PyArrayObject *detectfn_coefs = NULL;
      
      params.df_alpha = alpha;
      params.df_cos_beta = cos (beta);
      params.df_sin_beta = sin (beta);


      if (get_pyint_attr(detectfn, "kmax", &(params.df_kmax)) ||
	  get_pyint_attr(detectfn, "lmax", &(params.df_lmax)) ||
	  get_pybool_attr(detectfn, "oddl", &(df_oddl)) ||
	  get_pyfloat_attr(detectfn, "sigma", &(params.df_sigma)) ||
	  get_pyfloat_attr(detectfn, "rkstep", &(params.df_rkstep))
	  )
	return NULL;

	  params.df_two_sigma2 = 2.0 * params.df_sigma  * params.df_sigma;

      /* Grab the fit coefficients from the pbasex fit. We want to
	 store these as a normal C array, rather than having to access
	 them as a numpy array during the integration so we can
	 release the GIL. */
      p = PyObject_GetAttrString (detectfn, "coef");
      if (p == NULL)
	{
	  Py_XDECREF(p); /* Belt & braces */
	  return NULL;
	}

      /* Note the last argument here assures an aligned, contiguous array is returned. */
      detectfn_coefs = 
	(PyArrayObject *) PyArray_FROM_OTF(p, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
      
      if (detectfn_coefs == NULL)
	{
	  Py_XDECREF(p);
	  Py_XDECREF(detectfn_coefs);
	  return NULL;
	}
      
      /* Now we set up a pointer to the data array in
	 detectfn_coefs. Note that below we drop the GIL, but will
	 still continue to read data from this data array, itself
	 owned by a Python object. This could be problematic from a
	 GIL point of view, but this suggests not:
	 
	 http://stackoverflow.com/questions/8824739/global-interpreter-lock-and-access-to-data-eg-for-numpy-arrays
	 
	 The 100% safe alternative would be to copy the data to a
	 new memory location and use that once the GIL is dropped
	 below.
      */
      params.df_coefs = (double *) PyArray_DATA(detectfn_coefs);

      /* Sanity check */
      if (params.df_coefs == NULL || 
	  PyArray_SIZE(detectfn_coefs) != (params.df_lmax + 1) * (params.df_kmax + 1))
	{
	  Py_XDECREF(p);
	  Py_XDECREF(detectfn_coefs);
	  PyErr_SetString (PyExc_RuntimeError, "Failed to get a usable reference to detection function coefficients");
	  return NULL;
	}
    }

  /* Release GIL, as we're not accessing any Python objects for the
     main calculation. Any return statements will need to be preceeded
     by Py_BLOCK_THREADS - see /usr/include/pythonX/ceval.h */
  Py_BEGIN_ALLOW_THREADS

  fn.function = &integrand_detfn1;
  fn.params = &params;
  /* For speed in the integrand function we need to store the Legendre
     polynomials. But, we don't want to be malloc'ing and freeing
     storage in every call to the integrand function, so here we set
     upthe storage. To ensure minimal cache misses, when we're not
     considering odd l values, we close pack the Legendre
     polynomials.  */
  if (df_oddl)
    {
      df_ldim = params.df_lmax + 1;
      params.df_linc = 1;
    }
  else
    {
      df_ldim = params.df_lmax / 2 + 1;
      params.df_linc = 2;
    }

  params.df_legpol = malloc(df_ldim * sizeof(double));
  if (params.df_legpol == NULL)
    {
      Py_BLOCK_THREADS
      return PyErr_NoMemory();
    }

  matrix = malloc(Rbins * Thetabins * sizeof (double));
  if (matrix == NULL)
    {
      free (params.df_legpol);
      Py_BLOCK_THREADS
      return PyErr_NoMemory(); 
    }

  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  wksp = gsl_integration_workspace_alloc(wkspsize);
  if (!wksp)
    {
      free (matrix);
      free (params.df_legpol);
      Py_BLOCK_THREADS
      return PyErr_NoMemory();
    }

  table = gsl_integration_qaws_table_alloc(-0.5, 0.0, 0.0, 0.0);
  if (!table)
    {
      free (matrix);
      free (params.df_legpol);
      gsl_integration_workspace_free(wksp);
      Py_BLOCK_THREADS
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
	  params.RsinTheta = R * sin (Theta);
	  
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
	      matrix[R * Thetabins + j] = result;

	      /* Symmetry of Legendre polynomials is such that
		 P_L(cos(Theta))=P_L(cos(-Theta)), so we can exploit
		 that here unless Theta = 0 (which only occurs if
		 Thetabins is odd), in which case it's not needed.
	      */
	      if (!(ThetabinsOdd && j == midTheta))
		matrix[R * Thetabins + Thetabins - j - 1] = result;
	    }
	  else
	    {
	      char *errstring;

	      gsl_integration_workspace_free(wksp);
	      gsl_integration_qaws_table_free(table);
	      free (params.df_legpol);
	      free (matrix);

	      switch (status)
		{
		case GSL_EMAXITER:
		  if (asprintf(&errstring,
			       "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EROUND:
		  if (asprintf(&errstring,
			       "Failed to integrate: round-off error.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		case GSL_ESING:
		  if (asprintf(&errstring,
			       "Failed to integrate: singularity.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EDIVERGE:
		  if (asprintf(&errstring,
			       "Failed to integrate: divergent.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta))
		    errstring = NULL;
		  
		  break;
		  
		default:
		  if (asprintf(&errstring,
			       "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %d Theta: %f\n", 
			       status, k, l, R, Theta))
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
  free (params.df_legpol);

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
}


/****************************************************************
 * Python extension module initialization stuff.
 ****************************************************************/

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef MatrixMethods[] = {
    {"basisfn",  basisfn, METH_VARARGS,
     "Returns a matrix of a single (k, l) pBasex basis function."},
    {"basisfn_detfn1",  basisfn_detfn1, METH_VARARGS,
     "Returns a matrix of a single (k, l) pBasex basis function incorporating a detection function derived from a previous pBasex fit."},
    {"matrix",  matrix, METH_VARARGS,
     "Returns an inversion matrix of pBasex basis functions (deprecated)."},
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
