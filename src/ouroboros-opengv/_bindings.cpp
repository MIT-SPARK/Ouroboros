#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(_ouroboros_opengv, module) {
  module.def("foo", [](int a, int b) { return a + 2 * b; });
}
