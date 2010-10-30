/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <math.h>
#include <gsl/gsl_sf_legendre.h>

static inline int
arr1D_get(PyObject *arr, int idx, double *val)
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
arr1D_set(PyObject *arr, int idx, double val)
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
arr2D_get(PyObject *arr, int idx1, int idx2, double *val)
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
arr2D_set(PyObject *arr, int idx1, int idx2, double val)
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
  PyObject *coef = NULL, *coefarg = NULL;
  PyObject *dist;
  int rbins, thetabins, i, kmax, lmax;
  double rmax, rstep, thetastep, rkstep, sigma, s;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diiOiddi", 
			&rmax, &rbins, &thetabins, &coef, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to polar_distribution");
      return NULL;
    }

  coef = PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_IN_ARRAY);
  if (!coef)
    return NULL;

  dims[0] = (npy_intp) rbins;
  dims[1] = (npy_intp) thetabins;

  dist = PyArray_SimpleNew (2, dims, NPY_DOUBLE);
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

  return dist;

 fail:
  Py_DECREF(coef);
  Py_DECREF(dist);
  return NULL;
}

static PyObject *
beta_coeffs(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *coef = NULL;
  int rbins, i, kmax, lmax, ldim;
  double rmax, rstep, rkstep, sigma, s;
  PyObject *beta;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diOiddi", 
			&rmax, &rbins, &coefarg, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to beta_coeffs");
      return NULL;
    }

  coef = PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_IN_ARRAY);
  if (!coef)
    return NULL;

  ldim = lmax + 1;

  dims[0] = (npy_intp) ldim;
  dims[1] = (npy_intp) rbins;

  beta = PyArray_ZEROS (2, dims, NPY_DOUBLE, 0);
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

  return beta;

 fail:
  Py_DECREF(beta);
  Py_DECREF(coef);
  return NULL;
}

// TODO: speed up calculation by making use of mirror symmetry in y
static PyObject *
cartesian_distribution(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *coef = NULL;
  int npoints, i, kmax, lmax;
  double rmax, step, rkstep, sigma, s;
  PyObject *dist;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diOiddi", 
			&rmax, &npoints, &coefarg, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to cartesian_distribution");
      return NULL;
    }

  coef = PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_IN_ARRAY);
  if (!coef)
    return NULL;

  dims[0] = (npy_intp) npoints;
  dims[1] = (npy_intp) npoints;

  dist = PyArray_SimpleNew (2, dims, NPY_DOUBLE);
  if (!dist)
    {
      Py_DECREF(coef);
      return PyErr_NoMemory();
    }

  step = 2.0 * rmax / (npoints - 1);
  s = 2.0 * sigma * sigma;

  for (i = 0; i < npoints; i++)
    {
      double x = -rmax + i * step;
      int j;

      for (j = 0; j < npoints; j++)
	{
	  double y = -rmax + j * step;
	  double r = sqrt (x * x + y * y);
	  double val = 0.0;

	  if (r < rmax)
	    {
	      double costheta = cos(atan2(x, y));
	      int k;

	      for (k = 0; k <= kmax; k++)
		{
		  double rk = k * rkstep;
		  double a = r - rk;
		  double rad = exp(-(a * a) / s);
		  int l;
		  
		  for (l = 0; l <= lmax; l++)
		    {
		      double ang = gsl_sf_legendre_Pl(l, costheta);
		      double c;
		      
		      if (arr2D_get(coef, k, l, &c))
			goto fail;

		      val += c * rad * ang;
		    }
		}
	    }
	  if (arr2D_set(dist, i, j, val))
	    goto fail;
	}
    }

  Py_DECREF(coef);

  return dist;
  
 fail:
  Py_DECREF(coef);
  Py_DECREF(dist);
  return NULL;
}

static PyObject *
radial_spectrum(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *coef = NULL;
  PyObject *spec;
  int rbins, i, kmax;
  double rmax, rstep, rkstep, sigma, s, max = 0.0;
  npy_intp rbinsnp;

  if (!PyArg_ParseTuple(args, "diOidd", 
			&rmax, &rbins, &coefarg, &kmax, &rkstep, &sigma))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to radial_spectrum");
      return NULL;
    }

  coef = PyArray_FROM_OTF(coefarg, NPY_DOUBLE, NPY_IN_ARRAY);
  if (!coef)
    return NULL;

  /* spec = (double *) PyDataMem_NEW (rbins * sizeof (double)); */
  rbinsnp = (npy_intp) rbins;
  spec = PyArray_SimpleNew (1, &rbinsnp, NPY_DOUBLE);
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

  return spec;

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
    {"cartesian_distribution",  cartesian_distribution, METH_VARARGS,
     "Returns a simulated distribution cartesian image from fit coefficients."},
    {"polar_distribution",  polar_distribution, METH_VARARGS,
     "Returns a simulated distribution polar image from fit coefficients."},
    {"beta_coeffs",  beta_coeffs, METH_VARARGS,
     "Returns beta coefficents as a function of r from fit coefficients."},
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
}

