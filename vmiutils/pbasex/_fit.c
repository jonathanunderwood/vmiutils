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
	  goto fail;
	}

      if (PyArray_SETITEM(r_arr, idxp, valp))
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to set value of r_arr element");
	  goto fail;
	}

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
		  goto fail;
		}

	      if (PyArray_SETITEM(theta_arr, idxp, valp))
		{
		  PyErr_SetString (PyExc_RuntimeError, 
				   "Failed to set value of r_arr element");
		  goto fail;
		}
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
	      goto fail;
	    }

	  if (PyArray_SETITEM(dist, idxp, valp))
	    {
	      PyErr_SetString (PyExc_RuntimeError, 
			       "Failed to set value of dist element");
	      goto fail;
	    }

	}
    }

  retvalp = Py_BuildValue("OOO", r_arr, theta_arr, dist);
  return retvalp;

 fail:
  Py_DECREF(dist);
  Py_DECREF(r_arr);
  Py_DECREF(theta_arr);
  return NULL;
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
      PyErr_SetString (PyExc_TypeError, "Bad argument to calc_distribution");
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
	  goto fail;
	}

      if (PyArray_SETITEM(r_arr, idxp, valp))
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to set value of r_arr element");
	  goto fail;
	}

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
	  goto fail;
	}

      if (PyArray_SETITEM(spec, idxp, valp))
	{
	  PyErr_SetString (PyExc_RuntimeError, 
			   "Failed to set value of spec element");
	  goto fail;
	}

    }
  
  retvalp = Py_BuildValue("OO", r_arr, spec);
  return retvalp;

 fail:
  Py_DECREF(spec);
  Py_DECREF(r_arr);
  return NULL;
}

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef FitMethods[] = {
    {"calc_spectrum",  calc_spectrum, METH_VARARGS,
     "Returns a simulated angular integrated radial spectrum from fit coefficients."},
    {"calc_distribution",  calc_distribution, METH_VARARGS,
     "Returns a simulated distribution polar image from fit coefficients."},
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

