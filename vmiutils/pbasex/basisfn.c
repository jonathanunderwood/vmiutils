#include <Python.h>

static double integrand(r)
{
  return 0.0;
}

static PyObject *
basisfn(PyObject *self, PyObject *args)
{
  int k, l; /* Could use unsigned int here. */
  double R, Theta;

  if (!PyArg_ParseTuple(args, "i i d d", &k, &l, &R, &Theta))
    return NULL;

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

