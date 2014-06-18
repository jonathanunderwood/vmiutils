/* Note that Python.h must be included before any other header files. */
#include <Python.h>

/* For numpy we need to specify the API version we're targetting so
   that deprecated API warnings are issued when appropriate. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#undef NPY_NO_DEPRECATED_API

#include <math.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

/* Exceptions for this module. */
static PyObject *IntegrationError;

static inline int
arr1D_get(PyArrayObject *arr, int idx, double *val)
{
  PyObject *pval = PyArray_GETITEM(arr, PyArray_GETPTR1(arr, idx));
  if (pval)
    {
      *val = PyFloat_AsDouble(pval);
      Py_DECREF(pval);
      return 0;
    }
  else
    return -1;
}

static inline int
arr1D_set(PyArrayObject *arr, int idx, double val)
{
  PyObject *pval = PyFloat_FromDouble (val);
  if (pval)
    {
      int ret = PyArray_SETITEM(arr, PyArray_GETPTR1(arr, idx), pval);
      Py_DECREF(pval);
      return ret;
    }
  else
    return -1;
}

static inline int
arr2D_get(PyArrayObject *arr, int idx1, int idx2, double *val)
{
  PyObject *pval = PyArray_GETITEM(arr, PyArray_GETPTR2(arr, idx1, idx2));
  if (pval)
    {
      *val = PyFloat_AsDouble(pval);
      Py_DECREF(pval);
      return 0;
    }
  else
    return -1;
}

static inline int
arr2D_set(PyArrayObject *arr, int idx1, int idx2, double val)
{
  PyObject *pval = PyFloat_FromDouble (val);
  if (pval)
    {
      int ret = PyArray_SETITEM(arr, PyArray_GETPTR2(arr, idx1, idx2), pval);
      Py_DECREF(pval);
      return ret;
    }
  else
    return -1;
}

static PyObject *
polar_distribution(PyObject *self, PyObject *args)
{
  PyArrayObject *dist = NULL, *coef = NULL;
  PyObject *coefarg = NULL;
  int rbins, thetabins, i, kmax, lmax;
  double rmax, rstep, thetastep, rkstep, sigma, s;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diiOiddi",
			&rmax, &rbins, &thetabins, &coef, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to polar_distribution");
      return NULL;
    }

  coef = (PyArrayObject *) PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coef)
    return NULL;

  dims[0] = (npy_intp) rbins;
  dims[1] = (npy_intp) thetabins;

  dist = (PyArrayObject *) PyArray_SimpleNew (2, dims, NPY_DOUBLE);
  if (!dist)
    {
      Py_DECREF(coef);
      return PyErr_NoMemory();
    }

  rstep = rmax / (rbins - 1);
  thetastep = 2.0 * M_PI/ (thetabins - 1);
  s = 2.0 * sigma * sigma;

  for (i = 0; i < rbins; i++)
    {
      double r = i * rstep;
      int j;

      for (j = 0; j < thetabins; j++)
	{
	  double theta = j * thetastep;
	  int k;
	  double val = 0;

	  for (k = 0; k <= kmax; k++)
	    {
	      double rk = k * rkstep;
	      double a = r - rk;
	      double rad = exp(-(a * a) / s);
	      int l;

	      for (l = 0; l <= lmax; l++)
		{
		  double ang = gsl_sf_legendre_Pl(l, cos(theta));
		  double c;
		  if (arr2D_get(coef, k, l, &c))
		    goto fail;

		  val += c * rad * ang;
		}
	    }
	  if (arr2D_set(dist, i, j, val))
	    goto fail;
	}
    }

  Py_DECREF(coef);

  return (PyObject *) dist;

 fail:
  Py_DECREF(coef);
  Py_DECREF(dist);
  return NULL;
}

static PyObject *
beta_coeffs(PyObject *self, PyObject *args)
{
  PyArrayObject *coef = NULL, *beta = NULL;
  PyObject *coefarg = NULL;
  int rbins, i, kmax, lmax, ldim;
  double rmax, rstep, rkstep, sigma, s;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diOiddi",
			&rmax, &rbins, &coefarg, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to beta_coeffs");
      return NULL;
    }

  coef = (PyArrayObject *) PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coef)
    return NULL;

  ldim = lmax + 1;

  dims[0] = (npy_intp) ldim;
  dims[1] = (npy_intp) rbins;

  beta = (PyArrayObject *) PyArray_ZEROS (2, dims, NPY_DOUBLE, 0);
  if (!beta)
    {
      Py_DECREF (coef);
      return PyErr_NoMemory();
    }

  rstep = rmax / (rbins - 1);
  s = 2.0 * sigma * sigma;

  for (i = 0; i < rbins; i++)
    {
      double r = i * rstep;
      int k;

      for (k = 0; k <= kmax; k++)
	{
	  double rk = k * rkstep;
	  double a = r - rk;
	  double rad = exp(-(a * a) / s);
	  int l;

	  for (l = 0; l <= lmax; l++)
	    {
	      double b, c;
	      if (arr2D_get(coef, k, l, &c))
		goto fail;
	      if (arr2D_get(beta, l, i, &b))
		goto fail;
	      if (arr2D_set(beta, l, i, b + c * rad))
		goto fail;
	    }
	}
    }

  Py_DECREF(coef);

  /* Normalize to beta_0 = 1 at each r. */
  for (i = 0; i < rbins; i++)
    {
      double norm;
      int l;

      if (arr2D_get(beta, 0, i, &norm))
	goto fail;

      for (l = 0; l <= lmax; l++)
	{
	  double b;

	  if (arr2D_get(beta, l, i, &b))
	    goto fail;

	  if (arr2D_set(beta, l, i, b / norm))
	    goto fail;
	}
    }

  return (PyObject *) beta;

 fail:
  Py_DECREF(beta);
  Py_DECREF(coef);
  return NULL;
}

static PyObject *
cartesian_distribution_point(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *pycoef = NULL;
  int k, kmax, lmax, oddl, linc, kstart, kstop;
  double x, y, costheta, r, rkstep, sigma, s, val, truncate;
  double * coef;

  if (!PyArg_ParseTuple(args, "ddOiddiid",
			&x, &y, &coefarg, &kmax, &rkstep, &sigma, &lmax,
			&oddl, &truncate))
    {
      PyErr_SetString (PyExc_TypeError,
		       "Bad argument to cartesian_distribution_point");
      return NULL;
    }

  if (oddl == 1)
    linc = 1;
  else if (oddl == 0)
    linc = 2;
  else
    return NULL;

  pycoef = PyArray_FROM_OTF(coefarg, NPY_DOUBLE,
			    NPY_ARRAY_IN_ARRAY);
  if (!pycoef)
    {
      Py_XDECREF(pycoef);
      return NULL;
    }

  coef = (double *) PyArray_DATA((PyArrayObject *) pycoef);

  /* Release GIL here as no longer accessing python objects. */
  Py_BEGIN_ALLOW_THREADS

  s = 2.0 * sigma * sigma;
  r = sqrt (x * x + y * y);
  costheta = cos(atan2(x, y));

  kstart = floor((r - truncate * sigma) / rkstep);
  kstop = ceil((r + truncate * sigma) / rkstep);
  if (kstart < 0)
    kstart = 0;
  if (kstop > kmax)
    kstop = kmax;

  val = 0.0;
  for (k = kstart; k <= kstop; k++)
    {
      double rk = k * rkstep;
      double a = r - rk;
      double rad = exp(-(a * a) / s);
      int l;

      for (l = 0; l <= lmax; l += linc)
	{
	  double ang = gsl_sf_legendre_Pl(l, costheta);
	  double c = coef [k * (lmax + 1) + l];

	  val += c * rad * ang;
	}
    }

  /* Regain GIL */
  Py_END_ALLOW_THREADS

  Py_DECREF (pycoef);

  return Py_BuildValue("d", val);
}

/* Calculation of <cos^n theta> expectation values - this requires a
   numerical integration over theta. As such we need a structure to
   contain the integration parameters, and an integrand
   function. We'll use the GSL routines for the numerical
   integration. */
typedef struct
{
  int n;
  int lmax, linc;
  double *beta;
} cosn_expval_int_params;

static double
cosn_expval_integrand(double theta, void *params)
{
  cosn_expval_int_params p = *(cosn_expval_int_params *) params;
  double cos_theta = cos(theta), sin_theta = sin(theta);
  double cosn_theta = pow (cos_theta, p.n);
  double * beta = p.beta;
  double val = 0.0;
  int l;

  for (l = 0; l <= p.lmax; l += p.linc)
    val += beta[l] * gsl_sf_legendre_Pl(l, cos_theta);

  val *= cosn_theta * sin_theta;

  return val;
}

static PyObject *
cosn_expval_point(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *pycoef = NULL;
  int k, kmax, l, lmax, oddl, linc, kstart, kstop, n, nmax;
  double r, rkstep, sigma, s, truncate, norm, epsabs, epsrel;
  double *coef, *beta, *cosn;
  gsl_integration_workspace * wksp;
  size_t wkspsize=10000;
  gsl_function fn;
  cosn_expval_int_params params;
  npy_intp dims;
  PyObject *cosn_npy;

  if (!PyArg_ParseTuple(args, "diOiddiiddd",
			&r, &nmax, &coefarg, &kmax, &rkstep, &sigma, &lmax,
			&oddl, &truncate, &epsabs, &epsrel))
    {
      PyErr_SetString (PyExc_TypeError,
		       "Bad argument to cosn_expval_point");
      return NULL;
    }

  if (oddl == 1)
    linc = 1;
  else if (oddl == 0)
    linc = 2;
  else
    return NULL;

  pycoef = PyArray_FROM_OTF(coefarg, NPY_DOUBLE,
			    NPY_ARRAY_IN_ARRAY);
  if (!pycoef)
    {
      Py_XDECREF(pycoef);
      return NULL;
    }

  coef = (double *) PyArray_DATA((PyArrayObject *) pycoef);

  /* Release GIL here as no longer accessing python objects. */
  Py_BEGIN_ALLOW_THREADS

  /* Note we don't use calloc here, which only formally initializes
     int/char types to 0, and doesn't guarantee contiguous memory. */
  beta = malloc((lmax + 1) * sizeof(double));
  if (beta == NULL)
    {
      Py_BLOCK_THREADS
      Py_DECREF(pycoef);
      return PyErr_NoMemory();
    }

  for (l = 0; l <= lmax; l += linc)
    beta[l] = 0.0;

  /* Now we calculate the beta parameter values at this r value. */
  s = 2.0 * sigma * sigma;

  kstart = floor((r - truncate * sigma) / rkstep);
  kstop = ceil((r + truncate * sigma) / rkstep);
  if (kstart < 0)
    kstart = 0;
  if (kstop > kmax)
    kstop = kmax;

  for (k = kstart; k <= kstop; k++)
    {
      double rk = k * rkstep;
      double a = r - rk;
      double rad = exp(-(a * a) / s);
      int l;

      for (l = 0; l <= lmax; l += linc)
	{
	  double c = coef [k * (lmax + 1) + l];
	  beta[l] += c * rad;
	}
    }

  norm = beta[0];
  for (l = 0; l <= lmax; l += linc)
      beta[l] /= norm;

  /* Note we don't use calloc here, which only formally initializes
     int/char types to 0, and doesn't guarantee contiguous memory. */
  cosn = malloc((nmax + 1) * sizeof(double));
  if (cosn == NULL)
    {
      free (beta);
      Py_BLOCK_THREADS
      Py_DECREF (pycoef);
      return PyErr_NoMemory();
    }

  for (n = 0; n <= nmax; n++)
    cosn[n] = 0.0;

  params.lmax = lmax;
  params.linc = linc;
  params.beta = beta;

  wksp = gsl_integration_workspace_alloc (wkspsize);
  if (wksp == NULL)
    {
      free (beta);
      free(cosn);
      Py_BLOCK_THREADS
      Py_DECREF(pycoef);
      return PyErr_NoMemory();
    }

  fn.function = &cosn_expval_integrand;
  fn.params = &params;

  for (n = 0; n <= nmax; n += linc)
    {
      double result, abserr;
      int status;

      params.n = n;
      status = gsl_integration_qag (&fn, 0.0, M_PI, epsabs, epsrel,
				    wkspsize, 6, wksp, &result, &abserr);

      if (status == GSL_SUCCESS)
	  cosn[n] = result;
      else
	{
	  char *errstring;

	  switch (status)
	    {
	    case GSL_EMAXITER:
	      if (asprintf(&errstring,
			   "Failed to integrate: max number of subdivisions exceeded.\nn: %d\n", n) < 0)
		errstring = NULL;

	      break;

	    case GSL_EROUND:
	      if (asprintf(&errstring,
			   "Failed to integrate: round-off error.\nn: %d\n", n) < 0)
		errstring = NULL;

	      break;

	    case GSL_ESING:
	      if (asprintf(&errstring,
			   "Failed to integrate: singularity.\nn: %d\n", n) < 0)
		errstring = NULL;

	      break;

	    case GSL_EDIVERGE:
	      if (asprintf(&errstring,
			   "Failed to integrate: divergent.\nn: %d\n", n) < 0)
		errstring = NULL;

	      break;

	    default:
	      if (asprintf(&errstring,
			   "Failed to integrate: unknown error. status: %d.\nn: %d\n", status, n) < 0)
		errstring = NULL;

	      break;
	    }

	  gsl_integration_workspace_free(wksp);
	  free(beta);
	  free(cosn);

	  Py_BLOCK_THREADS
	    Py_DECREF(pycoef);
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

  gsl_integration_workspace_free (wksp);
  free(beta);

  norm = cosn[0];
  for (n = 0; n <= nmax; n += linc)
    cosn[n] /= norm;

  /* Regain GIL */
  Py_END_ALLOW_THREADS

  Py_DECREF (pycoef);

  dims = nmax + 1;
  cosn_npy = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, (void *) cosn);
  if (cosn_npy == NULL)
    {
      free (cosn);
      Py_XDECREF (cosn_npy); /* Belt and braces */
      return NULL;
    }

  /* Ensure the correct memory deallocator is called when Python
     destroys cosn_npy. In principle, it should be sufficient to
     just set the OWNDATA flag on cosn_npy, but this makes the
     assumption that numpy uses malloc and free for memory management,
     which it may not in the future. Note also that we are only able
     to use malloc and free because these are guaranteed to give
     aligned memory for standard types (which double is). For more
     complicated objects we'd need to ensure we have aligned alloc and
     free. See:
     http://blog.enthought.com/python/numpy/simplified-creation-of-numpy-arrays-from-pre-allocated-memory/
 */
  PyArray_SetBaseObject((PyArrayObject *) cosn_npy, PyCObject_FromVoidPtr(cosn, free));

  return cosn_npy;
}

static PyObject *
radial_spectrum(PyObject *self, PyObject *args)
{
  PyArrayObject *coef = NULL, *spec = NULL;
  PyObject *coefarg = NULL;
  int rbins, i, kmax;
  double rmax, rstep, rkstep, sigma, s, max = 0.0;
  npy_intp rbinsnp;

  if (!PyArg_ParseTuple(args, "diOidd",
			&rmax, &rbins, &coefarg, &kmax, &rkstep, &sigma))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to radial_spectrum");
      return NULL;
    }

  coef = (PyArrayObject *) PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!coef)
    return NULL;

  /* spec = (double *) PyDataMem_NEW (rbins * sizeof (double)); */
  rbinsnp = (npy_intp) rbins;
  spec = (PyArrayObject *) PyArray_SimpleNew (1, &rbinsnp, NPY_DOUBLE);
  if (!spec)
    {
      Py_DECREF(coef);
      return PyErr_NoMemory();
    }

  rstep = rmax / (rbins - 1);
  s = 2.0 * sigma * sigma;

  for (i = 0; i < rbins; i++)
    {
      double r = i * rstep;
      double val = 0.0;
      int k;

      for (k = 0; k <= kmax; k++)
	{
	  double rk = k * rkstep;
	  double a = r - rk;
	  double rad = exp(-(a * a) / s);
	  double c;

	  if (arr2D_get(coef, k, 0, &c))
	    goto fail;

	  val += c * rad * r * r;
	}

      if (arr1D_set(spec, i, val))
	goto fail;

      if (val > max)
	max = val;
    }

  Py_DECREF(coef);

  /* Normalize to maximum value of 1. */
  for (i = 0; i < rbins; i++)
    {
      double val;

      if (arr1D_get(spec, i, &val))
	goto fail;

      if (arr1D_set(spec, i, val / max))
	goto fail;
    }

  return (PyObject *) spec;

 fail:
  Py_DECREF(coef);
  Py_DECREF(spec);
  return NULL;
}

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef FitMethods[] = {
    {"radial_spectrum",  radial_spectrum, METH_VARARGS,
     "Returns a simulated angular integrated radial spectrum from fit coefficients."},
    {"cartesian_distribution_point",  cartesian_distribution_point, METH_VARARGS,
     "Returns a (x, y) point in the simulated distribution cartesian image from fit coefficients."},
    {"polar_distribution",  polar_distribution, METH_VARARGS,
     "Returns a simulated distribution polar image from fit coefficients."},
    {"beta_coeffs",  beta_coeffs, METH_VARARGS,
     "Returns beta coefficents as a function of r from fit coefficients."},
    {"cosn_expval_point", cosn_expval_point, METH_VARARGS,
     "Returns expectation values <cos^n theta> at a single value of r."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _basisfn. */
PyMODINIT_FUNC
init_fit(void)
{
  PyObject *mod;

  /* This is needed for the numpy API. */
  import_array();

  mod = Py_InitModule("_fit", FitMethods);
  if (mod == NULL)
    return;

  /* Exceptions. */
  IntegrationError = PyErr_NewException("_fit.IntegrationError", NULL, NULL);
  Py_INCREF(IntegrationError);
  PyModule_AddObject(mod, "IntegrationError", IntegrationError);
}
