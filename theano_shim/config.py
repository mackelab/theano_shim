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

