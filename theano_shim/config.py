"""
Global configuration variables
"""
import sys
# Make it safe to assume Python 3 when testing version
if sys.version_info.major < 3:
    raise RuntimeError("theano_shim requires Python 3. You are using {}."
                       .format(sys.version))
from collections import OrderedDict
if sys.version_info.minor >= 5:
    from typing import Union
import numpy as np

use_theano = False
inf = None

# Unified support for type hints; supported in Python >=3.5
if sys.version_info.minor >= 5:
    Numeric = Union[np.ndarray]

theano_updates = OrderedDict()
    # Stores a Theano update dictionary

lib = None
    # DEPRECATION WARNING: lib will soon be removed
RandomStreams = None

# TerminatingTypes is a tuple of all types which may be iterable but
# are treated as a single unit. This is used when recursively expanding
# an argument in core._expand_args.
# For example, if the sparse module is loaded, sparse types are added to TerminatingTypes,
# since we will never store a Theano object as an element of a SciPy sparse matrix
_TerminatingTypes = (str,)
