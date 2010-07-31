/* Note that Python.h must be included before any other header files. */
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_errno.h>

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
  double a, rad, ang;
  int_params p = *(int_params *) params;

  a = r - p.rk;
  rad = exp(-(a * a) / p.two_sigma2);
  ang = gsl_sf_legendre_Pl(p.l, p.RcosTheta / r);

  return r * rad * ang / sqrt(r * r - p.R2);
}

static PyObject *
basisfn_full(PyObject *self, PyObject *args)
{
  int l;
  double r, rk, sigma, theta, rad, ang, a, s;

  if (!PyArg_ParseTuple(args, "iidddd", &r, &rk, &sigma, &l, &theta));
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument");
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

  if (!PyArg_ParseTuple(args, "ddd", &r, &rk, &sigma));
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument");
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
  double sigma, epsabs, epsrel, tol; /* Suggest epsabs = 0.0, epsrel = tol = 1.0e-7 */   
  double rwidth, dTheta;
  int wkspsize; /* Suggest: wkspsize = 100000. */
  int ldim, kdim, midTheta, k;
  unsigned short int oddl, ThetabinsOdd, linc;
  npy_intp dims[4];
  PyObject *matrix;
  gsl_integration_workspace *wksp;
  gsl_function fn;
  int_params params;

  if (!PyArg_ParseTuple(args, "iiiidHdddi", 
			&kmax, &lmax, &Rbins, &Thetabins, &sigma, &oddl, &epsabs, &epsrel, &tol, &wkspsize))
    {
      PyErr_SetString (PyExc_TypeError, "Bad argument");
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

  fn.function = &integrand;
  fn.params = &params;

  rwidth = ((double) Rbins) / kdim;

  if (sigma < 0)
    sigma = rwidth / (2.0 * sqrt(2.0 * log(2.0)));

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
      params.rk = k * rwidth;
      for (l=0; l <= lmax; l+=linc)
	{
	  int R;
	  params.l = l;
	  for (R=0; R<Rbins; R++)
	    {
	      int j;
	      params.R2 = R * R;
	      for (j=0; j<=midTheta; j++)
		{
		  int status;
		  double result, abserr;
		  void *elementp;
		  PyObject *valp;
		  double Theta = -M_PI + j * dTheta;

		  params.RcosTheta = R * cos (Theta);

		  status = gsl_integration_qagiu (&fn, (double) R, epsabs, epsrel, wkspsize, 
						  wksp, &result, &abserr);

		  if (fabs(abserr / result) > tol)
		    {
		      PyErr_SetString (ToleranceError, "Failed to achieve desired integration tolerance");
		      goto fail;
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
		      
		      elementp = PyArray_GETPTR4(matrix, k, l, R, j);
		      if (!elementp)
			{
			  PyErr_SetString (PyExc_RuntimeError, "Failed to get pointer to matrix element");
			  goto fail;
			}
		      
		      if (PyArray_SETITEM(matrix, elementp, valp))
			{
			  PyErr_SetString (PyExc_RuntimeError, "Failed to set value of matrix element");
			  goto fail;
			}
			
		      /* Symmetry of Legendre polynomials is such that
			 P_L(cos(Theta))=P_L(cos(-Theta)), so we can exploit
			 that here unless Theta = 0, in which case it's not
			 needed. */
		      if (ThetabinsOdd && j == midTheta)
			continue;

		      elementp = PyArray_GETPTR4(matrix, k, l, R, Thetabins - j - 1);
		      if (!elementp)
			{
			  PyErr_SetString (PyExc_RuntimeError, "Failed to get pointer to matrix element");
			  goto fail;
			}
		      
		      if (PyArray_SETITEM(matrix, elementp, valp))
			{
			  PyErr_SetString (PyExc_RuntimeError, "Failed to set value of matrix element");
			  goto fail;
			}
		      break;

		    case GSL_EMAXITER:
		      PyErr_SetString (MaxIterError, 
				       "Maximum number of integration subdivisions exceeded");
		      goto fail;
		      
		    case GSL_EROUND:
		      PyErr_SetString (RoundError, "Failed to achieve required integration tolerance");
		      goto fail;
		      
		    case GSL_ESING:
		      PyErr_SetString (SingularError, "Failed to integrate: singularity found");
		      goto fail;
		      
		    case GSL_EDIVERGE:
		      PyErr_SetString (DivergeError, "Failed to integrate: divergent or slowly convergent");
		      goto fail;
		      
		    default:
		      PyErr_SetString (PyExc_RuntimeError, "Failed to integrate: Unknown error");
		      goto fail;
		    }	
		}

	    }
	}
    }

  gsl_integration_workspace_free(wksp);

  return matrix;

 fail:  
  gsl_integration_workspace_free(wksp);
  Py_DECREF(matrix);
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
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _basisfn. */
PyMODINIT_FUNC
init_basisfn(void)
{
  PyObject *mod;

  /* This is needed for the nump API. */
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

