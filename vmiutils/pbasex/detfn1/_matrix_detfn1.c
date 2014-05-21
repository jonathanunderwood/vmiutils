      
/****************************************************************
 * Structures and functions for pBasex incorporating a detection
 * function specified as a previous pBasex fit.
 ****************************************************************/

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

/* This factor determines how many Gaussian sigmas in r we consider
   when setting the upper bound in the integration. */
#define __UPPER_BOUND_FACTOR 10.0

/* This factor determines how many detection function basis functions
   either side of r to consider when calculating the integrand. It
   probably makes most sense to set this the same as
   __UPPER_BOUND_FACTOR above. */
#define __DF_N_SIGMA 10.0


/* Exceptions for this module. */
static PyObject *IntegrationError;

typedef enum
{
  qags, qaws, cquad
} integration_method;

typedef struct 
{
  int l, df_linc, df_kmax, df_lmax;
  double R, rk, two_sigma2, RcosTheta, RsinTheta;
  double threshold;
  double df_sigma, df_two_sigma2, df_rkstep, df_rmax;
  double df_alpha, df_cos_beta, df_sin_beta;
  double *df_coef, *df_beta;
  integration_method method;
} int_params_detfn1;

static double 
integrand_detfn1 (double r, void *params)
/* Integrand for PBASEX with a detection function defined by a
   previous PBASEX fit. Here we calculate the detection frame theta
   angle using Zare Eq 3.86 from theta and phi in the lab frame, and
   the angles alpha and beta between the lab frame and detection
   frame.

   In theory the angle phi in the code below could be calculated like this:

        double phi = asin (p.RsinTheta / (r * sin_theta));

   However, in practice, lack of precision in the numerator and
   denominator causes a problem when the argument should evaluate to
   -1.0, it ends up evaluating to slightly less than -1.0, and asin
   returns NaN. And so below we have to work around this by testing if
   sin_phi lies outside the range [-1,1], and if so, force it back
   into that range. This is very ugly, and can mask other problems. I
   wish there was a better way!
*/
{
  int_params_detfn1 p = *(int_params_detfn1 *) params;
  double cos_theta = p.RcosTheta / r;
  double sin_theta = sqrt ((1.0 + cos_theta) * (1.0 - cos_theta));/* sin (acos(cos_theta)) */
  double sin_phi = p.RsinTheta / (r * sin_theta);
  double phi, cos_theta_det_frame;
  double a, rad, ang, val, df_val, delta;
  int k, l, df_lidx, kmin, kmax;

  if (sin_phi < -1.0)
    phi = -M_PI / 2.0;
  else if (sin_phi > 1.0)
    phi = M_PI;
  else
    phi = asin (sin_phi);

  cos_theta_det_frame = p.df_cos_beta * cos_theta + 
    p.df_sin_beta * sin_theta * cos (p.df_alpha - phi);

  /* Usual basis function for pbasex */
  a = r - p.rk;
  rad = exp (-(a * a) / p.two_sigma2);
  ang = gsl_sf_legendre_Pl (p.l, cos_theta);

  if (p.method == qaws)
    val = rad * ang * r / sqrt(r + p.R);
  else /* p.method == qags || p.method == cquad */
    val = rad * ang * r / sqrt((r + p.R) * (r - p.R));

  /* Now calculate the angular detection function. Here the strategy
     is to calulcate the Beta coefficients from the detection function
     pbasex coefficients at this value of r and use the angular part
     of the fit only - i.e. we do not include the radial part of the
     detection function fit. When calculating the Beta parameters at
     r, we only consider basis functions which are a maximum delta
     away from this value of r. */
  delta = __DF_N_SIGMA * p.df_sigma;
  kmin = floor((r - delta) / p.df_rkstep);
  kmax = ceil((r + delta) / p.df_rkstep);
  if (kmin < 0)
    kmin = 0;
  if (kmax > p.df_kmax)
    kmax = p.df_kmax;

  df_lidx = -1;
  for (l = 0; l <= p.df_lmax; l += p.df_linc)
    {
      df_lidx ++;
      p.df_beta[l] = 0.0;
    }

  /* Calculate beta parameters. */
  for (k = kmin; k <= kmax; k++)
    {	      
      double rk = k * p.df_rkstep;
      double a = r - rk;
      double rad = exp(-(a * a) / p.df_two_sigma2);

      df_lidx = -1;

      for (l = 0; l <= p.df_lmax; l += p.df_linc)
	{
	  df_lidx++;
	  p.df_beta[df_lidx] += rad * p.df_coef[k * (p.df_lmax + 1) + l];
	}
    }

  /* Calculate angular part of the detection function distribution at
     this R, taking care to normalize to Beta_0=1. */
  df_val = 0.0;
  df_lidx = -1;
  for (l = 0; l <= p.df_lmax; l += p.df_linc)
    {
      df_lidx ++;
      p.df_beta[df_lidx] /= p.df_beta[0]; /* Normalize to Beta_0 = 1 */
      df_val += p.df_beta[df_lidx] * gsl_sf_legendre_Pl(l, cos_theta_det_frame); 
    }

  val = df_val * val;

  /* Round small values to 0 otherwise the integration error becomes
     dominated by the numerical error in exp such that the relative
     error is huge and the integration fails. Also, for qaws and qags,
     we need to ensure that NAN is not returned which happens when
     when R=r=0.0. cquad is designed to handle nan values. Be careful
     here with future changes to the code, as this switching of nan
     for 0.0 could hide other problems. Note also that here any
     occurences of inf will be set to 0.0 for all methods - at present
     these don't occur, but any changes to the code should be careful
     of this. If in doubt, comment out what follows and examine all
     values being returned. */
  if (isnan(val))
    {
      if (p.method == cquad)
	return val;
      else
	return 0.0;
    }

  if (fabs(val) > p.threshold)
    return val;
  else
    return 0.0;
}

static PyObject *
basisfn_detfn1(PyObject *self, PyObject *args)
/* Calculates a single (k, l) PBASEX basis function and returns it as
   a two dimensional Numpy array Python object. This includes a
   detection function specified as a previous PBASEX fit. */
{
  int Rbins, Thetabins, k, l, R;
  int wkspsize; /* Suggest: wkspsize = 100000. */
  int midTheta, df_oddl;
  unsigned short int ThetabinsOdd;
  int df_kmax, df_lmax;
  double df_sigma, df_rkstep, df_alpha, df_beta;
  double sigma, epsabs, epsrel; /* Suggest epsabs = 0.0, epsrel = 1.0e-7 */   
  double rk, dTheta, upper_bound, threshold;
  double *matrix = NULL;
  void *wksp;
  gsl_function fn;
  gsl_integration_qaws_table *table = NULL;
  int_params_detfn1 params;
  integration_method method;
  char * method_str;
  npy_intp dims[2];
  PyObject *numpy_matrix, *df_coef_arg;
  PyArrayObject *df_coef = NULL;
  size_t df_ldim;

  if (!PyArg_ParseTuple(args, "iiiiddddidOiddiidds",
			&k, &l, &Rbins, &Thetabins, &sigma, &rk,
			&epsabs, &epsrel, &wkspsize, &threshold,
			&df_coef_arg, &df_kmax, &df_sigma,
			&df_rkstep, &df_lmax, &df_oddl,
			&df_alpha, &df_beta, &method_str
			))
    {
      PyErr_SetString (PyExc_TypeError, "Bad arguments to basisfn_detfn1");
      return NULL;
    }

  /* Grab the fit coefficients from the pbasex fit. We want to store
     these as a normal C array, rather than having to access them as a
     numpy array during the integration so we can release the
     GIL. Note the last argument here assures an aligned, contiguous
     array is returned. Note that we can't decref df_coef_arg here, as
     it seems that this function returns a stolen reference (in other
     words, decref'ing df_coef_arg does causes segfault. */
  df_coef =
    (PyArrayObject *) PyArray_FROM_OTF(df_coef_arg, NPY_DOUBLE, 
				       NPY_ARRAY_IN_ARRAY);

  if (df_coef == NULL)
    {
      Py_XDECREF(df_coef);
      return NULL;
    }

  /* Now we set up a pointer to the data array in detectfn_coefs. Note
     that below we drop the GIL, but will still continue to read data
     from this data array, itself owned by a Python object. This could
     be problematic from a GIL point of view, but this suggests not:
	 
     http://stackoverflow.com/questions/8824739/global-interpreter-lock-and-access-to-data-eg-for-numpy-arrays
	 
     The 100% safe alternative would be to copy the data to a new
     memory location and use that once the GIL is dropped below.

     Note that we can't decref df_coef until we have finished
     accessing its data via our C pointer.
  */
  params.df_coef = (double *) PyArray_DATA(df_coef);

  if (params.df_coef == NULL)
    {
      Py_DECREF(df_coef);
      return NULL;
    }

  /* Release GIL, as we're not accessing any Python objects for the
     main calculation. Any return statements will need to be preceeded
     by Py_BLOCK_THREADS - see /usr/include/pythonX/ceval.h */
  Py_BEGIN_ALLOW_THREADS

  params.two_sigma2 = 2.0 * sigma * sigma;
  params.rk = rk;
  params.l = l;
  params.df_lmax = df_lmax;
  params.df_kmax = df_kmax;
  params.df_sigma = df_sigma;
  params.df_two_sigma2 = 2.0 * df_sigma  * df_sigma;
  params.df_rkstep = df_rkstep;
  params.df_alpha = df_alpha;
  params.df_cos_beta = cos (df_beta);
  params.df_sin_beta = sin (df_beta);
  params.threshold = threshold;
  
  /* method_str should be one of 'qags', 'qaws', or 'cquad' */
  if (!strncmp(method_str, "qaws", 4))
    method = qaws;
  else if (!strncmp(method_str, "qags", 4))
    method = qags;
  else
    method = cquad;

  params.method = method;
  /* For speed in the integrand function we need to store the Legendre
     polynomials. But, we don't want to be malloc'ing and freeing
     storage in every call to the integrand function, so here we set
     up the storage. To ensure minimal cache misses, when we're not
     considering odd l values, we close pack the Legendre
     polynomials.  */

  if (df_oddl)
    {
      df_ldim = df_lmax + 1;
      params.df_linc = 1;
    }
  else
    {
      df_ldim = df_lmax / 2 + 1;
      params.df_linc = 2;
    }

  params.df_beta = malloc(df_ldim * sizeof (double));
  if (params.df_beta == NULL)
    {
      Py_BLOCK_THREADS
      Py_DECREF(df_coef);
      return PyErr_NoMemory();
    }

  matrix = malloc(Rbins * Thetabins * sizeof (double));
  if (matrix == NULL)
    {
      free (params.df_beta);
      Py_BLOCK_THREADS
      Py_DECREF(df_coef);
      return PyErr_NoMemory(); 
    }

  fn.function = &integrand_detfn1;
  fn.params = &params;

  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  if (method == cquad)
    wksp = gsl_integration_cquad_workspace_alloc(wkspsize);
  else
    wksp = gsl_integration_workspace_alloc(wkspsize);

  if (!wksp)
    {
      free (matrix);
      free (params.df_beta);
      Py_BLOCK_THREADS
      return PyErr_NoMemory();
    }

  if (method == qaws)
    {
      table = gsl_integration_qaws_table_alloc(-0.5, 0.0, 0.0, 0.0);
      if (!table)
	{
	  free (matrix);
	  free (params.df_beta);
	  gsl_integration_workspace_free(wksp);
	  Py_BLOCK_THREADS
	  Py_DECREF(df_coef);
	  return PyErr_NoMemory();
	}
    }

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

  upper_bound = params.rk + __UPPER_BOUND_FACTOR * sigma;
  
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
	      switch (method)
		{
		case qaws:
		  status = gsl_integration_qaws (&fn, (double) R, upper_bound, table,
						 epsabs, epsrel, wkspsize,
						 wksp, &result, &abserr);
		  break;
		case qags:
		  status = gsl_integration_qags (&fn, (double) R, upper_bound,
						 epsabs, epsrel, wkspsize,
						 wksp, &result, &abserr);
		  break;
		case cquad:
		  status = gsl_integration_cquad (&fn, (double) R, upper_bound,
						  epsabs, epsrel,
						  wksp, &result, &abserr, 
						  (size_t *) &wkspsize);
		  break;
		}
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
	      switch (method)
		{
		case cquad:
		  gsl_integration_cquad_workspace_free(wksp);
		  break;
		case qaws:
		  gsl_integration_qaws_table_free(table);
		  /* No break intentionally */
		case qags:
		  gsl_integration_workspace_free(wksp);
		  break;
		}

	      free (params.df_beta);
	      free (matrix);

	      switch (status)
		{
		case GSL_EMAXITER:
		  if (asprintf(&errstring,
			       "Failed to integrate: max number of subdivisions exceeded.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EROUND:
		  if (asprintf(&errstring,
			       "Failed to integrate: round-off error.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		case GSL_ESING:
		  if (asprintf(&errstring,
			       "Failed to integrate: singularity.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		case GSL_EDIVERGE:
		  if (asprintf(&errstring,
			       "Failed to integrate: divergent.\nk: %d l: %d R: %d Theta: %f\n", 
			       k, l, R, Theta) < 0)
		    errstring = NULL;
		  
		  break;
		  
		default:
		  if (asprintf(&errstring,
			       "Failed to integrate: unknown error. status: %d.\nk: %d l: %d R: %d Theta: %f\n", 
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
	      Py_DECREF(df_coef);
	      
	      return NULL;
	    }
	}
    }

  switch (method)
    {
    case cquad:
      gsl_integration_cquad_workspace_free(wksp);
      break;
    case qaws:
      gsl_integration_qaws_table_free(table);
      /* No break intentionally */
    case qags:
      gsl_integration_workspace_free(wksp);
      break;
    }

  free (params.df_beta);

  /* Regain GIL as we'll now access some Python objects. */
  NPY_END_ALLOW_THREADS

  Py_DECREF(df_coef);

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
    {"basisfn_detfn1",  basisfn_detfn1, METH_VARARGS,
     "Returns a matrix of a single (k, l) pBasex basis function incorporating a detection function derived from a previous pBasex fit."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _matrix_detfn1. */
PyMODINIT_FUNC
init_matrix_detfn1(void)
{
  PyObject *mod;

  /* This is needed for the numpy API. */
  import_array();

  mod = Py_InitModule("_matrix_detfn1", MatrixMethods);
  if (mod == NULL)
    return;

  /* Exceptions. */
  IntegrationError = PyErr_NewException("_matrix_detfn1.IntegrationError", NULL, NULL);
  Py_INCREF(IntegrationError);
  PyModule_AddObject(mod, "IntegrationError", IntegrationError);
}

#undef __UPPER_BOUND_FACTOR 
#undef __DF_N_SIGMA
