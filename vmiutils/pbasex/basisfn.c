#include <Python.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>

/* Size of workspace used for numerical integration. */
#define WKSPSIZE 1000
#define EPSABS 1.0e-8
#define EPSREL 1.0e-8

typedef struct 
{
  int k, l;
  double R2, RcosTheta, rk, sigma2;
} int_params;

static void
int_params_init (int_params *p, int k, int l, double R, double Theta,
		 double rk, double sigma)
{
  p->k = k;
  p->l = l;
  p->R2 = R * R;
  p->rk = rk;
  p->sigma2 = sigma * sigma;
}

static double integrand(double r, void *params)
{
  double a, rad, ang;
  int_params p = *(intparams *) params;

  a = r - p.rk;
  rad = exp(-(a * a) / p.sigma2);
  ang = gsl_sf_legendre_Pl(p.l, p.RcosTheta / r);

  return r * rad * ang / sqrt(r * r - R2);
}

static PyObject *
basisfn(PyObject *self, PyObject *args)
{
  int k, l; /* Could use unsigned int here. */
  double R, Theta, rk, sigma;
  gsl_integration_workspace wksp;
  gsl_function fn;
  int_params params;

  if (!PyArg_ParseTuple(args, "i i d d d d", &k, &l, &R, &Theta, rk, sigma))
    return NULL;

  int_params_init (&params, k, l, R, Theta, rk, sigma);
  wksp = gsl_integration_workspace_alloc(WKSPSIZE);
  fn.function = &integrand;
  fn.params = &params;
  
  result = gsl_integration_qagiu (fn, R, epsabs, epsrel, WKSPSIZE,
  gsl_integration_workspace_free(&wksp);

}
static PyMethodDef BasisFnMethods[] = {
    {"basisfn",  basisfn, METH_VARARGS,
     "Calculate the value of a basis function."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initbasisfn(void)
{
    (void) Py_InitModule("basisfn", BasisFnMethods);
}

