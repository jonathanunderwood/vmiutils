/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <math.h>
#include <gsl/gsl_sf_legendre.h>

static PyObject *
polar_distribution(PyObject *self, PyObject *args)
{
  PyObject *coef = NULL, *coefarg = NULL;
  PyObject *distnp = NULL;
  int rbins, thetabins, i, kmax, lmax, index = -1;
  double rmax, rstep, thetastep, rkstep, sigma, s;
  double *dist = NULL;
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

  dist = malloc (rbins * thetabins * sizeof (double));
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
	  double val = 0.0;
	  int k;

	  index++;

	  for (k = 0; k <= kmax; k++)
	    {
	      double rk = k * rkstep;
	      double a = r - rk;
	      double rad = exp(-(a * a) / s);
	      int l;

	      for (l = 0; l <= lmax; l++)
		{
		  double ang = gsl_sf_legendre_Pl(l, cos(theta));
		  double *cvalp;

		  cvalp = (double *) PyArray_GETPTR2(coef, k, l);
		  if (!cvalp)
		    {
		      PyErr_SetString (PyExc_RuntimeError, 
				       "Failed to get pointer to coefficient");
		      Py_DECREF(coef);
		      free (dist);
		      return NULL;
		    }

		  // TODO: should probably be using PyArray_GETVAL here
		  val += (*cvalp) * rad * ang;
		  Py_DECREF(cvalp);
		}
	    }
	  dist[index] = val;
	}
    }

  dims[0] = (npy_intp) rbins;
  dims[1] = (npy_intp) thetabins;

  distnp = PyArray_SimpleNewFromData (2, dims, NPY_DOUBLE, dist);
  if (!distnp)
    {
      free(dist);
      Py_DECREF(coef);
      return PyErr_NoMemory();
    }

  Py_DECREF(coef);

  return distnp;
}

static PyObject *
beta_coeffs(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *coef = NULL;
  int rbins, i, kmax, lmax, ldim;
  double rmax, rstep, rkstep, sigma, s;
  double *beta = NULL;
  PyObject *betanp = NULL;
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

  beta = calloc (rbins * ldim, sizeof (double));
  if (!beta)
    {
      Py_DECREF(coef);
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
	      double *cvalp;
	      int index = l * rbins + i; /* i.e. beta[l, r] */

	      cvalp = (double *) PyArray_GETPTR2(coef, k, l);
	      if (!cvalp)
		{
		  PyErr_SetString (PyExc_RuntimeError, 
				   "Failed to get pointer to coefficient");
		  Py_DECREF(coef);
		  free (beta);
		  return NULL;
		}
	      
	      // TODO: should probably be using PyArray_GETVAL here
	      beta[index] += (*cvalp) * rad;
	      Py_DECREF(cvalp);
	    }
	}
    }

  Py_DECREF(coef);

  /* Normalize to beta_0 = 1 at each r. */
  for (i = 0; i < rbins; i++)
    {
      double norm = beta[i]; /* i.e. beta[0, r] */
      int l;

      for (l = 0; l <= lmax; l++)
	{
	  int index = l * rbins + i; /* i.e. beta[l, r] */
	  /* if (norm > 0.0) */
	    beta[index] /= norm;
	  /* else */
	  /*   beta[index] = 0.0; */
	}
    }

  dims[0] = (npy_intp) ldim;
  dims[1] = (npy_intp) rbins;

  betanp = PyArray_SimpleNewFromData (2, dims, NPY_DOUBLE, beta);
  if (!beta)
    {
      free(beta);
      return PyErr_NoMemory();
    }

  return betanp;
}

// TODO: speed up calculation by making use of mirror symmetry in y
static PyObject *
cartesian_distribution(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *coef = NULL;
  int npoints, i, kmax, lmax, index = -1;
  double rmax, step, rkstep, sigma, s;
  double *dist;
  PyObject *distnp;
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

  dist = malloc (npoints * npoints * sizeof (double));
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
	  int k;

	  index++;

	  if (r < rmax)
	    {
	      double costheta = cos(atan2(x, y));

	      for (k = 0; k <= kmax; k++)
		{
		  double rk = k * rkstep;
		  double a = r - rk;
		  double rad = exp(-(a * a) / s);
		  int l;
		  
		  for (l = 0; l <= lmax; l++)
		    {
		      double ang = gsl_sf_legendre_Pl(l, costheta);
		      double *cvalp = NULL;
		      
		      cvalp = (double *) PyArray_GETPTR2(coef, k, l);
		      if (!cvalp)
			{
			  PyErr_SetString (PyExc_RuntimeError, 
					   "Failed to get pointer to coefficient");
			  Py_DECREF(coef);
			  free (dist);
			  return NULL;
			}
		      
		      // TODO: should probably be using PyArray_GETVAL here
		      printf ("%d %d %g %g\n", k, l, *cvalp, 0.0/50.0);
		  
		      val += (*cvalp) * rad * ang;
		      Py_DECREF(cvalp);
		    }
		}
	    }
	  dist[index] = val;
	}
    }

  Py_DECREF(coef);

  dims[0] = (npy_intp) npoints;
  dims[1] = (npy_intp) npoints;

  distnp = PyArray_SimpleNewFromData (2, dims, NPY_DOUBLE, dist);
  if (!distnp)
    {
      free(dist);
      return PyErr_NoMemory();
    }

  return distnp;
}

static PyObject *
radial_spectrum(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *coef = NULL;
  PyObject *specnp;
  int rbins, i, kmax;
  double rmax, rstep, rkstep, sigma, s, max = 0.0;
  double *spec;
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

  spec = malloc (rbins * sizeof (double));
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
	  double *cvalp;

	  cvalp = (double *) PyArray_GETPTR2(coef, k, 0);
	  if (!cvalp)
	    {
	      PyErr_SetString (PyExc_RuntimeError, 
			       "Failed to get pointer to coefficient");
	      Py_DECREF(coef);
	      free(spec);
	      return NULL;
	    }

	  // TODO: should probably be using PyArray_GETVAL here
	  val += (*cvalp) * rad * r * r;
	  Py_DECREF(cvalp);
	}
      spec[i] = val;
      if (val > max)
	max = val;
    }

  Py_DECREF(coef);

  /* Normalize to maximum value of 1. */
  for (i = 0; i < rbins; i++)
    spec[i] /= max;

  rbinsnp = (npy_intp) rbins;

  specnp = PyArray_SimpleNewFromData (1, &rbinsnp, NPY_DOUBLE, spec);
  if (!specnp)
    {
      free (spec);
      return PyErr_NoMemory();
    }

  return specnp;
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

