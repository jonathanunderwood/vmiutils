
#include <Python.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_errno.h>

typedef struct 
{
  int l;
  double R2, RcosTheta, rk, two_sigma2;
} int_params;

static void
int_params_init (int_params *p, const double R, const double Theta,
		 const int l, const double rk, const double sigma)
{
  p->l = l;
  p->R2 = R * R;
  p->rk = rk;
  p->two_sigma2 = 2.0 * sigma * sigma;
  p->RcosTheta = R * cos(Theta);
}

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
basisfn(PyObject *self, PyObject *args)
{
  int l, status, wkspsize; /* A sensible choice for wkspsize is 100000. */
  double R, Theta, rk, sigma, result, abserr;
  double epsabs, epsrel; /* Note: a sensible choice is epsabs = 0.0 */  
  gsl_integration_workspace *wksp;
  gsl_function fn;
  int_params params;

  if (!PyArg_ParseTuple(args, "ddiddddi", 
			&R, &Theta, &l, &rk, &sigma, &epsabs, &epsrel, &wkspsize))
    return NULL;

  int_params_init (&params, R, Theta, l, rk, sigma);

  /* Turn off gsl error handler - we'll check return codes. */
  gsl_set_error_handler_off ();

  wksp = gsl_integration_workspace_alloc(wkspsize);
  if (!wksp)
    {
      PyErr_SetString (PyExc_MemoryError, 
		       "Failed to allocate workspace for integration workspace");
      return NULL;
    }

  fn.function = &integrand;
  fn.params = &params;
  
  status = gsl_integration_qagiu (&fn, R, epsabs, epsrel, wkspsize, wksp, &result, &abserr);
  gsl_integration_workspace_free(wksp);

  switch (status)
    {
    case GSL_SUCCESS:
      return Py_BuildValue("d d", result, abserr);

    case GSL_EMAXITER:
      PyErr_SetString (PyExc_RuntimeError, 
		       "Maximum number of integration subdivisions exceeded");
      return NULL;

    case GSL_EROUND:
      PyErr_SetString (PyExc_RuntimeError, 
		       "Failed to achieve required integration tolerance");
      return NULL;

    case GSL_ESING:
      PyErr_SetString (PyExc_RuntimeError, "Failed to integrate: singularity found");
      return NULL;

    case GSL_EDIVERGE:
      PyErr_SetString (PyExc_RuntimeError, "Failed to integrate: divergent or slowly convergent");
      return NULL;
    }
}

/* Module function table. Each entry specifies the name of the function exported
   by the module and the corresponding C function. */
static PyMethodDef BasisFnMethods[] = {
    {"basisfn",  basisfn, METH_VARARGS,
     "Calculate the value of a basis function."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization function, must be caled initNAME, where NAME is the
   compiled module name, in this case _basisfn. */
PyMODINIT_FUNC
init_basisfn(void)
{
    (void) Py_InitModule("_basisfn", BasisFnMethods);
}

