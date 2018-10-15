# Description:
#   Build rule for Python and Numpy.
#   This rule works for Debian and Ubuntu. Other platforms might keep the
#   headers in different places, cf. 'How to build DeepMind Lab' in build.md.

cc_library(
    name = "python",
    hdrs = glob([
        "PYINCDIR/*.h",
        "PYLIBDIR/site-packages/numpy/core/include/numpy/*.h",
    ]),
    includes = [
        "PYINCDIR",
        "PYLIBDIR/site-packages/numpy/core/include",
    ],
    visibility = ["//visibility:public"],
)
