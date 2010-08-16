/* TODO: 
   Use a better type than unsigned short for boolenas
   Use epsabs, rather than setting val=o in integrand for small values
   Call basisfn_full from integrand rather than duplicate code. Also make
   integrand_full inline for efficiency. Or make it a macro.
*/

/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_errno.h>

#define __SMALL 1.0e-30
#define __UPPER_BOUND_FACTOR 20.0

/* Exceptions for this module. */
static PyObject *MaxIterError;
static PyObject *RoundError;
static PyObject *SingularError;
static PyObject *DivergeError;
static PyObject *ToleranceError;

typedef struct 
{
  int l;
  double R2, RcosTheta, rk, two_sigma2;
} int_params;

static double integrand(double r, void *params)
{
  double a, rad, ang, val;
  int_params p = *(int_params *) params;

  a = r - p.rk;
  rad = exp (-(a * a) / p.two_sigma2);
  ang = gsl_sf_legendre_Pl (p.l, p.RcosTheta / r);

  val = r * rad * ang;

  if (fabs(val) > __SMALL)
    return val;
  else 
    return 0.0;
}

static PyObject *
basisfn_full(PyObject *self, PyObject *args)
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
basisfn_radial(PyObject *self, PyObject *args)
{
  double r, rk, sigma, rad, a, s;

  if (!PyArg_ParseTuple(args, "ddd", &r, &rk, &sigma))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument to basisfn_radial");
      return NULL;
    }

  a = r - rk;
  s = 2.0 * sigma * sigma;
  rad = exp(-(a * a) / s);

  return Py_BuildValue("d", rad);
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
	
  /* Create numpy array to hold the matrix. */
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
      Py_DECREF(matrix);
      return PyErr_NoMemory();
    }

  table = gsl_integration_qaws_table_alloc(-0.5, 0.0, 0.0, 0.0);
  if (!table)
    {
      gsl_integration_workspace_free(wksp);
      Py_DECREF(matrix);
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
	      params.R2 = (double) (R * R);
	      for (j = 0; j <= midTheta; j++)
		{
		  int status;
		  double result, abserr;
		  void *elementp;
		  PyObject *valp;
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
		      status = 0;
		      result = 0.0;
		      abserr = 0.0;
		    }

		  switch (status)
		    {
		    case GSL_SUCCESS:
		      valp = Py_BuildValue("d", result);
		      if (!valp)
			{
			  PyErr_SetString (PyExc_RuntimeError, 
					   "Failed to create python object for matrix element value");
			  goto fail;
			}
		      
		      if (oddl)
			elementp = PyArray_GETPTR4(matrix, k, l, R, j);
		      else
			elementp = PyArray_GETPTR4(matrix, k, l  / 2, R, j);

		      if (!elementp)
			{
			  PyErr_SetString (PyExc_RuntimeError, 
					   "Failed to get pointer to matrix element");
			  goto fail;
			}
		      
		      if (PyArray_SETITEM(matrix, elementp, valp))
			{
			  PyErr_SetString (PyExc_RuntimeError, 
					   "Failed to set value of matrix element");
			  goto fail;
			}
			
		      /* Symmetry of Legendre polynomials is such that
			 P_L(cos(Theta))=P_L(cos(-Theta)), so we can exploit
			 that here unless Theta = 0, in which case it's not
			 needed. */
		      if (ThetabinsOdd && j == midTheta)
			continue;

		      if (oddl)
			elementp = PyArray_GETPTR4(matrix, k, l, R, Thetabins - j - 1);
		      else
			elementp = PyArray_GETPTR4(matrix, k, l / 2, R, Thetabins - j - 1);

		      if (!elementp)
			{
			  PyErr_SetString (PyExc_RuntimeError, 
					   "Failed to get pointer to matrix element");
			  goto fail;
			}
		      
		      if (PyArray_SETITEM(matrix, elementp, valp))
			{
			  PyErr_SetString (PyExc_RuntimeError, 
					   "Failed to set value of matrix element");
			  goto fail;
			}
		      break;

		    case GSL_EMAXITER:
		      PyErr_SetString (MaxIterError, 
				       "Maximum number of integration subdivisions exceeded");
		      goto fail;
		      
		    case GSL_EROUND:
		      printf("k:%d l:%d R:%d Theta: %f\n", k, l, R, Theta);
		      PyErr_SetString (RoundError, "Failed to integrate: round-off error");
		      goto fail;
		      
		    case GSL_ESING:
		      printf("k:%d l:%d R:%d Theta: %f\n", k, l, R, Theta);
		      PyErr_SetString (SingularError, "Failed to integrate: singularity found");
		      goto fail;
		      
		    case GSL_EDIVERGE:
		      printf("k:%d l:%d R:%d Theta: %f\n", k, l, R, Theta);
		      PyErr_SetString (DivergeError, "Failed to integrate: divergent or slowly convergent");
		      goto fail;
		      
		    default:
		      printf("k:%d l:%d R:%d Theta: %f\n", k, l, R, Theta);
		      PyErr_SetString (PyExc_RuntimeError, "Failed to integrate: Unknown error");
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
	  double val = 0;
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
  return NULL;
}

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef BasisFnMethods[] = {
    {"basisfn_full",  basisfn_full, METH_VARARGS,
     "Returns the value of a basis function."},
    {"basisfn_radial",  basisfn_radial, METH_VARARGS,
     "Returns the value of the radial part of a basis function."},
    {"matrix",  matrix, METH_VARARGS,
     "Returns an inversion matrix of basis functions."},
    {"calc_distribution",  calc_distribution, METH_VARARGS,
     "Returns a simulated distribution from fit coefficients."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _basisfn. */
PyMODINIT_FUNC
init_basisfn(void)
{
  PyObject *mod;

  /* This is needed for the numpy API. */
  import_array();

  mod = Py_InitModule("_basisfn", BasisFnMethods);
  if (mod == NULL)
    return;

  /* Exceptions. */
  MaxIterError = PyErr_NewException("_basisfn.MaxIterError", NULL, NULL);
  Py_INCREF(MaxIterError);
  PyModule_AddObject(mod, "MaxIterError", MaxIterError);

  RoundError = PyErr_NewException("_basisfn.RoundError", NULL, NULL);
  Py_INCREF(RoundError);
  PyModule_AddObject(mod, "RoundError", RoundError);

  SingularError = PyErr_NewException("_basisfn.SingularError", NULL, NULL);
  Py_INCREF(SingularError);
  PyModule_AddObject(mod, "SingularError", SingularError);

  DivergeError = PyErr_NewException("_basisfn.DivergeError", NULL, NULL);
  Py_INCREF(DivergeError);
  PyModule_AddObject(mod, "DivergeError", DivergeError);

  ToleranceError = PyErr_NewException("_basisfn.ToleranceError", NULL, NULL);
  Py_INCREF(ToleranceError);
  PyModule_AddObject(mod, "ToleranceError", ToleranceError);
}

#undef __SMALL
#undef __UPPER_BOUND_FACTOR 
