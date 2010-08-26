/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <math.h>
#include <gsl/gsl_sf_legendre.h>

static PyObject *
calc_distribution(PyObject *self, PyObject *args)
{
  PyObject *coef;
  PyObject *dist, *r_arr, *theta_arr, *retvalp;
  int rbins, thetabins, i, kmax, lmax;
  double rmax, rstep, thetastep, rkstep, sigma, s;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diiOiddi", 
			&rmax, &rbins, &thetabins, &coef, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to calc_distribution");
      return NULL;
    }

  dims[0] = (npy_intp) rbins;
  dims[1] = (npy_intp) thetabins;

  dist = PyArray_SimpleNew (2, dims, NPY_DOUBLE);
  if (!dist)
    return PyErr_NoMemory();

  r_arr = PyArray_SimpleNew (1, &(dims[0]), NPY_DOUBLE);
  if (!r_arr)
    {
      Py_DECREF(dist);
      return PyErr_NoMemory();
    }

  theta_arr = PyArray_SimpleNew (1, &(dims[1]), NPY_DOUBLE);
  if (!theta_arr)
    {
      Py_DECREF(dist);
      Py_DECREF(r_arr);
      return PyErr_NoMemory();
    }

  rstep = rmax / (rbins - 1);
  thetastep = 2.0 * M_PI/ (thetabins - 1);
  s = 2.0 * sigma * sigma;

  for (i = 0; i < rbins; i++)
    {
      double r = i * rstep;
      int j;
      PyObject *valp;
      void *idxp;

      valp = Py_BuildValue("d", r);
      if (!valp)
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to create python object for r value");
	  goto fail;
	}

      idxp = PyArray_GETPTR1(r_arr, i);
      if (!idxp)
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to get pointer to r_arr element");
	  Py_DECREF(valp);
	  goto fail;
	}

      if (PyArray_SETITEM(r_arr, idxp, valp))
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to set value of r_arr element");
	  Py_DECREF(valp);
	  Py_DECREF(idxp);
	  goto fail;
	}

      Py_DECREF(valp);
      Py_DECREF(idxp);

      for (j = 0; j < thetabins; j++)
	{
	  double theta = j * thetastep;
	  double val = 0.0;
	  int k;

	  if (i == 0)
	    {
	      valp = Py_BuildValue("d", theta);
	      if (!valp)
		{
		  PyErr_SetString (PyExc_RuntimeError, 
				   "Failed to create python object for theta value");
		  goto fail;
		}

	      idxp = PyArray_GETPTR1(theta_arr, j);
	      if (!idxp)
		{
		  PyErr_SetString (PyExc_RuntimeError, 
				   "Failed to get pointer to theta_arr element");
		  Py_DECREF(valp);
		  goto fail;
		}

	      if (PyArray_SETITEM(theta_arr, idxp, valp))
		{
		  PyErr_SetString (PyExc_RuntimeError, 
				   "Failed to set value of r_arr element");
		  Py_DECREF(valp);
		  Py_DECREF(idxp);
		  goto fail;
		}
	      Py_DECREF(valp);
	      Py_DECREF(idxp);
	    }

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
		      goto fail;
		    }

		  val += (*cvalp) * rad * ang;
		  Py_DECREF(cvalp);
		}
	    }

	  valp = Py_BuildValue("d", val);
	  if (!valp)
	    {
	      PyErr_SetString (PyExc_RuntimeError, 
			       "Failed to create python object for dist value");
	      goto fail;
	    }

	  idxp = PyArray_GETPTR2(dist, i, j);
	  if (!idxp)
	    {
	      PyErr_SetString (PyExc_RuntimeError, 
			       "Failed to get pointer to dist element");
	      Py_DECREF(valp);
	      goto fail;
	    }

	  if (PyArray_SETITEM(dist, idxp, valp))
	    {
	      PyErr_SetString (PyExc_RuntimeError, 
			       "Failed to set value of dist element");
	      Py_DECREF(idxp);
	      Py_DECREF(valp);
	      goto fail;
	    }

	}
    }

  retvalp = Py_BuildValue("NNN", r_arr, theta_arr, dist);
  return retvalp;

 fail:
  Py_DECREF(dist);
  Py_DECREF(r_arr);
  Py_DECREF(theta_arr);
  return NULL;
}

static PyObject *
calc_distribution2(PyObject *self, PyObject *args)
{
  PyObject *coef;
  int rbins, thetabins, i, kmax, lmax, index = -1;
  double rmax, rstep, thetastep, rkstep, sigma, s;
  double *dist;
  PyObject *distnp;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diiOiddi", 
			&rmax, &rbins, &thetabins, &coef, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to calc_distribution2");
      return NULL;
    }

  dist = malloc (rbins * thetabins * sizeof (double));
  if (!dist)
    return PyErr_NoMemory();

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
      return PyErr_NoMemory();
    }

  return distnp;
}

static PyObject *
beta_coeffs(PyObject *self, PyObject *args)
{
  PyObject *coef;
  int rbins, i, kmax, lmax, ldim;
  double rmax, rstep, rkstep, sigma, s;
  double *beta;
  PyObject *betanp;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diOiddi", 
			&rmax, &rbins, &coef, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to beta_coefs");
      return NULL;
    }

  ldim = lmax + 1;

  beta = calloc (rbins * ldim, sizeof (double));
  if (!beta)
    return PyErr_NoMemory();

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
		  free (beta);
		  return NULL;
		}
	      
	      // TODO: should probably be using PyArray_GETVAL here
	      beta[index] += (*cvalp) * rad;
	      Py_DECREF(cvalp);
	    }
	}
    }

  /* Normalize to beta_0 = 1 at each r. */
  for (i = 0; i < rbins; i++)
    {
      double r = i * rstep;
      double norm = beta[i]; /* i.e. beta[0, r] */
      int l;

      for (l = 0; l <= lmax; l++)
	{
	  int index = l * rbins + i; /* i.e. beta[l, r] */
	  beta[index] /= norm;
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
  PyObject *coef;
  int npoints, i, kmax, lmax, index = -1;
  double rmax, step, rkstep, sigma, s;
  double *dist;
  PyObject *distnp;
  npy_intp dims[2];

  if (!PyArg_ParseTuple(args, "diOiddi", 
			&rmax, &npoints, &coef, &kmax, &rkstep, &sigma, &lmax))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to calc_distribution2");
      return NULL;
    }

  dist = malloc (npoints * npoints * sizeof (double));
  if (!dist)
    return PyErr_NoMemory();

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
	  double theta = atan2(x, y);
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
calc_spectrum(PyObject *self, PyObject *args)
{
  PyObject *coef;
  PyObject *spec, *r_arr, *retvalp;
  int rbins, i, kmax;
  double rmax, rstep, rkstep, sigma, s;
  npy_intp rbins_np;

  if (!PyArg_ParseTuple(args, "diOidd", 
			&rmax, &rbins, &coef, &kmax, &rkstep, &sigma))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to calc_spectrum");
      return NULL;
    }

  rbins_np = (npy_intp) rbins;

  spec = PyArray_SimpleNew (1, &rbins_np, NPY_DOUBLE);
  if (!spec)
    return PyErr_NoMemory();

  r_arr = PyArray_SimpleNew (1, &rbins_np, NPY_DOUBLE);
  if (!r_arr)
    {
      Py_DECREF(spec);
      return PyErr_NoMemory();
    }

  rstep = rmax / (rbins - 1);
  s = 2.0 * sigma * sigma;

  for (i = 0; i < rbins; i++)
    {
      double r = i * rstep;
      int k;
      PyObject *valp;
      void *idxp;
      double val = 0.0;

      valp = Py_BuildValue("d", r);
      if (!valp)
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to create python object for r value");
	  goto fail;
	}

      idxp = PyArray_GETPTR1(r_arr, i);
      if (!idxp)
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to get pointer to r_arr element");
	  Py_DECREF(valp);
	  goto fail;
	}

      if (PyArray_SETITEM(r_arr, idxp, valp))
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to set value of r_arr element");
	  Py_DECREF(idxp);
	  Py_DECREF(valp);
	  goto fail;
	}
      Py_DECREF(idxp);
      Py_DECREF(valp);

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
	      goto fail;
	    }

	  val += (*cvalp) * rad * r * r;
	  Py_DECREF(cvalp);
	}

      valp = Py_BuildValue("d", val);
      if (!valp)
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to create python object for spec value");
	  goto fail;
	}

      idxp = PyArray_GETPTR1(spec, i);
      if (!idxp)
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to get pointer to spec element");
	  Py_DECREF(valp);
	  goto fail;
	}

      if (PyArray_SETITEM(spec, idxp, valp))
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to set value of spec element");
	  Py_DECREF(idxp);
	  Py_DECREF(valp);
	  goto fail;
	}
      Py_DECREF(idxp);
      Py_DECREF(valp);
    }
  
  retvalp = Py_BuildValue("NN", r_arr, spec);
  return retvalp;

 fail:
  Py_DECREF(spec);
  Py_DECREF(r_arr);
  return NULL;
}

static PyObject *
calc_spectrum2(PyObject *self, PyObject *args)
{
  PyObject *coef;
  PyObject *specnp;
  int rbins, i, kmax;
  double rmax, rstep, rkstep, sigma, s, max = 0.0;
  double *spec;
  npy_intp rbinsnp;

  if (!PyArg_ParseTuple(args, "diOidd", 
			&rmax, &rbins, &coef, &kmax, &rkstep, &sigma))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to calc_spectrum2");
      return NULL;
    }

  spec = malloc (rbins * sizeof (double));
  if (!spec)
    return PyErr_NoMemory();

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
    {"calc_spectrum",  calc_spectrum, METH_VARARGS,
     "Returns a simulated angular integrated radial spectrum from fit coefficients."},
    {"calc_spectrum2",  calc_spectrum2, METH_VARARGS,
     "Returns a simulated angular integrated radial spectrum from fit coefficients."},
    {"calc_distribution",  calc_distribution, METH_VARARGS,
     "Returns a simulated distribution polar image from fit coefficients."},
    {"cartesian_distribution",  cartesian_distribution, METH_VARARGS,
     "Returns a simulated distribution cartesian image from fit coefficients."},
    {"calc_distribution2",  calc_distribution2, METH_VARARGS,
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

