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

/* Note that Python.h must be included before any other header files. */
#include <Python.h>

/* For numpy we need to specify the API version we're targetting so
   that deprecated API warnings are issued when appropriate. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#undef NPY_NO_DEPRECATED_API

#include <math.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_errno.h>


static PyObject *
detfn_cartesian_distribution_point(PyObject *self, PyObject *args)
{
  PyObject *coefarg = NULL, *pycoef = NULL;
  int k, kmax, l, lmax, oddl, linc, kstart, kstop;
  double x, y, beta, costheta, cosbeta, r, rkstep, sigma, s, val, truncate;
  double * coef, * ang;

  if (!PyArg_ParseTuple(args, "dddOiddiid",
			&x, &y, &beta, &coefarg, &kmax, &rkstep,
			&sigma, &lmax, &oddl, &truncate))
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

  cosbeta = cos(beta);

  ang = malloc((lmax + 1) * sizeof(double));
  if (ang == NULL)
    {
      Py_BLOCK_THREADS
      Py_XDECREF(pycoef);
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for ang in detfn_cartesian_distribution_point");
      return NULL;
    }

  for (l = 0; l <= lmax; l += linc)
    {
      ang[l] = gsl_sf_legendre_Pl(l, costheta) * gsl_sf_legendre_Pl(l, cosbeta);
    }

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

      for (l = 0; l <= lmax; l += linc)
	{
	  val += coef [k * (lmax + 1) + l] * rad * ang[l];
	}
    }

  free(ang);

  /* Regain GIL */
  Py_END_ALLOW_THREADS

  Py_DECREF (pycoef);

  return Py_BuildValue("d", val);
}


/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef FitDetFn1Methods[] = {
    {"detfn_cartesian_distribution_point",  detfn_cartesian_distribution_point, METH_VARARGS,
     "Returns a (x, y) point in the simulated distribution cartesian image for the detection function from detection function fit coefficients."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _fit_detfn1. */
PyMODINIT_FUNC
init_fit_detfn1(void)
{
  PyObject *mod;

  /* This is needed for the numpy API. */
  import_array();

  mod = Py_InitModule("_fit_detfn1", FitDetFn1Methods);
  if (mod == NULL)
    return;

}
