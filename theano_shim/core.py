"""
A unified interface for Numpy and Theano, so we don't need
conditionals just to select between e.g. T.sum and np.sum.
More specific calls can be dealt with in the related code by
conditioning on this module's `use_theano` flag

This module provides an interchangeable interface to common operations,
such as type casting and checking, assertions and rounding, as well
as 'shim' datatypes for random number streams and shared variables.

Usage
-----
At the top of your code, include the line
``import theano_shim as shim``
By default this will not even try to load Theano, so you can use it on
a machine where Theano is not installed.
To 'switch on' Theano, add the following below the import:
``shim.load('theano')``
You can switch it back to its default state with ``shim.load('numpy')``.


Pointers for writing theano switches
------------------------------------
- Type checking
    + isinstance(x, theano.tensor.TensorVariable) will be True when
      x is a theano variable, but False for wrappers around Python
      objects such as shared variables.
    + isinstance(x, theano.graph.basic.Variable) is more inclusive, returning
      True for shared variables as well.
    + These two tests are provided by the `is_theano_variable` and
      `is_theano_object` convenience methods.
"""

import logging
import builtins
from collections import OrderedDict
from collections.abc import Sequence, Iterable
from numbers import Number
import inspect
import sys
import copy as copymodule

import numpy as np
import scipy as sp
import scipy.signal
import scipy.special

from .config import config
from .config import _gettheano, _getT
cf = config

logger = logging.getLogger('theano_shim')
#logger.setLevel(logging.INFO)


######################
def load_theano():
    load('theano')

class DummyT:
    def __getattr__(self, attr):
        global T
        if not cf.use_theano:
            raise RuntimeError("Tried to access theano.tensor, but Theano has not been loaded.")
        else:
            assert not isinstance(T, DummyT)
            return getattr(T, attr)
T = DummyT()

def load(library='theano', reraise=False):
    """Reset the module to use or not use Theano.
    This should be called once at the top of your code.

    Parameters
    ----------
    library: 'numpy' | 'theano'
        - 'numpy'   : Module will simulate Theano using pure Numpy.
                      This is the state before calling `load()`.
        - 'theano'  : Module will act as an interface to Theano
    reraise: Boolean
        If true, import errors will be reraised to allow them to propagate to the parent.
    """
    #TODO: Move as much as possible to config.Config
    #      And/or move the function to config, and import config.load

    global theano, T

    if library == 'theano':
        try:
            import theano
        except ImportError:
            logger.error("The theano library was not found.")
            cf.library = 'numpy'
            if reraise:
                raise
        else:
            cf.library = 'theano'
    else:
        cf.library = library  # Raises error if `library` is invalid

    if cf.floatX == 'float32':
        config.make_constants_32bit()

    if cf.use_theano:
        import theano.ifelse
        import theano.tensor as T
        import theano.tensor.signal.conv
        import theano.sparse
        import theano.sandbox
        #import theano.tensor.shared_RandomStream  # CPU only
        # from theano.tensor.random.utils import RandomStream
        # import theano.sandbox.rng_mrg
        from theano.sandbox.rng_mrg import MRG_RandomStream as RandomStream  # CPU & GPU
        # The theano-pymc docs now recommend MRG_RandomStream, and that seems
        # to be where the development effort is. For example,
        # `RandomStream().binomial(n=1, p=0.9)` fails with “unknown keyword argument 'n'
        # but `MRG_RandomStream().binomial(n=1, p=0.9)` works fine

        from . import theano_types

        cf.add_terminating_types((T.TensorType, T.TensorVariable))

        cf.inf = 1e12
        # cf.RandomStream = \
        #     make_TheanoRNG(theano.tensor.shared_RandomStream.RandomStream)
        cf.RandomStream = theano_types.MRG_RNG

        # if cf.sys.version_info.minor >= 5:
        #     cf.Numeric = cf.Union[np.ndarray, T.TensorVariable]

    else:
        cf.inf = np.inf
        cf.RandomStream = NumpyRNG

        # if cf.sys.version_info.minor >= 5:
        #     cf.Numeric = cf.Union[np.ndarray]

    # Call the `reload()` methods in the submodules
    from . import graph
    graph.reload()

def gettheano():
    if not cf.use_theano:
        raise RuntimeError("Tried to access theano, but it has not been loaded.")
    else:
        return theano
def getT():
    if not cf.use_theano:
        raise RuntimeError("Tried to access theano.tensor, but Theano has not been loaded.")
    else:
        return T

class LazyEval:
    """
    Small wrapper to permit lazy evaluation of arguments.
    Python by default evaluates every argument being passed to a function,
    which can lead to problems e.g. when using ifelse as a guard:
        a = ifelse( neq(x, 0), y/x, y )
    In this case we can rewrite the above as
        a = ifelse( neq(x, 0), LazyEval(lambda x,y: y/x, (x,y)), y )

    Current functions compatible with LazyEval:
        - ifelse
    """
    def __init__(self, f, args=()):
        """
        Parameters
        ----------
        f: callable
            An expression which returns the desired value
        args: tuple
            The variables appearing in the function f.
        """
        self.f = f
        self.args = args
    def eval(self):
        return self.f(*self.args)

##########################
# Querying the computational graph
# (Moved to graph.py)

def is_computable(varlist, with_inputs=None):
    logger.warning("Deprecation warning: theano_shim.graph.is_computable() is deprecated. "
                   "Use theano_shim.graph.is_computable(). This test has NOT been executed.")

##########################
# Managing theano updates

def add_update(variable, value=None):
    """
    Parameters
    ----------
    variable: shared variable | dict | iterable
        Shared variable to update.
        Can also be a dictionary of `variable:value` pairs, or an iterable of
        `(variable, value)` tuples.
    value: symbolic expression
        Value to assign to variable. Ignored if `variable` is a dict or iterable

    Returns
    -------
    None
    """
    if isinstance(variable, dict):
        for key, val in variable.items():
            add_update(key, val)
    elif isinstance(variable, Sequence):
        for key, val in variable:
            add_update(key, val)
    else:
        logger.debug(f"Adding Theano update: {variable.name} -> {str(value)}")
        if not isshared(variable):
            raise ValueError("The updates mechanism only applies to shared variables.")
        cf.symbolic_updates[variable] = value
add_updates = add_update

def remove_update(variable):
    """
    Parameters
    ----------
    variable: shared variable | dict | iterable
        Shared variable to update.
        Can also be a dictionary of `variable:value` pairs, or an iterable of
        `(variable, value)` tuples.

    Returns
    -------
    None
    """
    if isinstance(variable, dict):
        for key, val in variable.items():
            add_update(key, val)
    elif isinstance(variable, Sequence):
        for key, val in variable:
            add_update(key, val)
    else:
        logger.debug(f"Removing Theano update: {variable.name}")
        if not isshared(variable):
            raise ValueError("The updates mechanism only applies to shared variables.")
        del cf.symbolic_updates[variable]
remove_updates = remove_update

def get_updates():
    return cf.symbolic_updates

def reset_updates():
    logger.debug("Clearing Theano updates")
    cf.symbolic_updates = OrderedDict()

def pending_update(*args):
    """
    Return True if there is a pending symbolic updates for any one of the
    variables in `args`.
    If called with no arguments, return True if the update dictionary is nonempty.
    """
    if len(args) == 0:
        return len(cf.symbolic_updates) > 0
    else:
        for x in _expand_args(args):
            if is_graph_object(x) and x in cf.symbolic_updates:
                return True
        return False
pending_updates = pending_update

#######################
# Print statement
def _get_print_fn(file=sys.stdout):
    """Return the same function as theano.printing._print_fn, with
    the difference that 'file' is passed as a keyword argument to print().
    """
    def _print_fn(op, xin,):
        for attr in op.attrs:
            temp = getattr(xin, attr)
            if callable(temp):
                pmsg = temp()
            else:
                pmsg = temp
            print(op.message, attr, '=', pmsg, file=file)
    return _print_fn

def print(x, message=None, printfn='print', message_prefix="SHIM - ",
          file=sys.stdout):
    """
    Non-Theano version outputs to the logger at the debug level.

    Parameters
    ----------
    x:
        The value of this graph will be output
    message: string
        Will be prepended to the output. If unspecified, the function
        checks if `x` has a `name` attribute and uses it if present.
    printfn: string
        Determines the function used to print the variable; only
        has an effect on Theano variables. Possible values are:
        - 'print' (default): use theano.printing.Print
        - 'debugprint': use theano.printing.debugprint
        - 'eval' : try to call x's `eval` method. If successful,
          print the output, otherwise fall back on theano.printing.Print
    message_prefix: string
        String to prepend to the message. Can be used to distinguish
        different types of outputs. Defaults to "SHIM - ".
    file: file handle
        Where to print the value; default is 'sys.stdout'.
        Same argument as used in print() or theano.printing.debugprint.
    """
    if message is None:
        message = getattr(x, 'name', "")
        if message is None: message = ""  # x.name might be None
    if is_theano_object(x):
        # EARLY EXIT: - slice
        # TODO?: list, tuple
        if isinstance(x, slice):
            kw = dict(printfn=printfn, message_prefix=message_prefix, file=file)
            start = print(x.start, message=message+" (start)", **kw)
            stop = print(x.stop, message=message+" (stop)", **kw)
            step = x.step and print(x.step, message=message+" (step)", **kw)
            return slice(start, stop, step)
        msg = message_prefix + message
        if printfn == 'print':
            return theano.printing.Print(msg, global_fn=_get_print_fn(file))(x)
        elif printfn == 'debugprint':
            builtins.print(msg)
            theano.printing.debugprint(x, file=file)
            return x
        elif printfn == 'eval':
            try:
                val = x.eval()
            except theano.graph.fg.MissingInputError:
                return theano.printing.Print(msg, global_fn=_get_print_fn(file))(x)
            else:
                builtins.print(msg + " Value of {}: {}".format(str(x), val))
                return x
        else:
            raise ValueError("Unrecognized print flag '{}'."
                             .format(printfn))
    else:
        if len(message) > 0 and message[-1] != " ":
            msg = message + " "
        else:
            msg = message
        #logger.debug(msg + str(x))
        builtins.print(message_prefix + msg + str(x), file=file)
        return x

def print_array(x, idx=slice(None), message=None, message_prefix="SHIM - ",
                file=sys.stdout):
    """
    Helper function for printing just one element in an array.
    All parameters except `idx` are the same as for `print`.
    Returns an identity operation on `x`, so that it can be used as follows

    >>> x = shim.tensor(np.arange(100, 0.1))
    >>> x = shim.print_array(x, idx=3)
    >>> for i in range(2):
    >>>   x *= shim.print_array(x, idx=np.s_[2:5])
    0.3__str__ DEBUG -
    [0.4, 0.6, 0.8]__str__ DEBUG -
    [0.8, 1.2, 1.6]__str__ DEBUG -
    """
    return set_subtensor(x[idx],
                         print(x[idx],
                               message=message,
                               message_prefix=message_prefix,
                               file=file
                               )
                         )

def pprint(x):
    """
    Call pretty printer (`pprint`) on Theano objects, otherwise standard `print`
    """
    if is_theano_object(x):
        return _gettheano().printing.pprint(x)
    else:
        return str(x)

def debugprint(x, file=sys.stdout):
    return print(x, printfn='debugprint', message="", message_prefix="", file=file)

#######################
# Assert equivalent
def check(stmt):
    """
    FIXME: In its current state this function is essentially broken, because
          even for statements like `x == y`, with x,y Theano variables, the
          *statement* is still a pure Python object, and this a plain assert
          gets used (and then usually fails).
    Check is a library-aware wrapper for assert.
    If stmt is a Theano variable, the behaviour depends on whether
    theano.config.compute_test_value:
        - If it is 'off', `check` is a no-op
        - Otherwise, use the test values to evaluate the assert
    """
    if not cf.use_theano or not isinstance(stmt, theano.graph.basic.Variable):
        assert(stmt)
    else:
        if theano.config.compute_test_value == 'off':
            return None
        else:
            assert(stmt.tag.test_value)

######################
# Retrieving test values
def get_test_value(var, nofail=False):
    """
    If `value` is a Theano variable, return its test value if it is defined.
    Otherwise just return `value` unchanged.
    If `nofail` is False (default), will raise an error if no test value is found.
    Otherwise returns None
    """
    if 'theano' in sys.modules and isinstance(var, _getT().sharedvar.SharedVariable):
        retval = var.get_value()
    elif 'theano' in sys.modules and isinstance(var, _gettheano().graph.basic.Variable):
        try:
            retval = var.tag.test_value
        except AttributeError:
            if nofail:
                return None
            else:
                raise AttributeError("You've attempted to execute a function that "
                                     "requires a test_value for the variable {} to "
                                     "be set, and this value is not set.".format(var))
    else:
        retval = var
    return retval

######################
# Type checking
def istype(obj, type_str):
    """
    Parameters
    ----------
    obj: object
        The object of which we want to check the type.
    type_str: string or dtype or iterable
        If `obj` is of this type, the function returns True,
        otherwise it returns False. Valid values of `type_str`
        are those expected for a dtype. Examples are:
        - 'int', 'int32', 'uint', 'uint32', etc.
        - 'float', 'float32', etc.
        - any NumPy dtype
        `type_str` can also be an iterable of aforementioned
        strings. Function will return True if `obj` is of any
        of the specified types

    Returns
    -------
    bool
    """
    # Wrap type_str if it was not passed as an iterable
    if isinstance(type_str, str) or not isinstance(type_str, Iterable):
        type_str = [type_str]
    # Ensure we only have strings (not dtypes)
    type_str = [str(ts) for ts in type_str]
    # Check type
    if ('theano' not in sys.modules
        or not isinstance(obj, _gettheano().graph.basic.Variable)):
        return any(ts in str(np.asarray(obj).dtype) for ts in type_str)
            # We cast to string to be consistent with Theano, which uses
            # strings for it's dtypes
    else:
        return any(ts in obj.dtype for ts in type_str)

def _expand_args(arglst):
    """
    Recursively expand slices, iterables, dictionaries into a list of scalar data type.
    Scalars are returned as a 1 element list.
    """
    if not isinstance(arglst, Iterable):
        arglst = [arglst]
    elif ('theano' in sys.modules
          and isinstance(arglst, _gettheano().graph.basic.Variable)):
        arglst = [arglst]
    elif isinstance(arglst, cf.TerminatingTypes):
        arglst = [arglst]
    for arg in arglst:
        if 'theano' in sys.modules and isinstance(arg, _gettheano().graph.basic.Variable):
            # Theano variables aren't iterable
            yield arg
        elif isinstance(arg, cf.TerminatingTypes):
            yield arg
        elif isinstance(arg, slice):
            yield arg.start
            yield arg.stop
            yield arg.step
        elif isinstance(arg, dict):
            for key in arg.keys():
                yield key
            for val in arg.values():
                #yield from nwlst.extend(_expand_args(val))
                yield from _expand_args(val)
        elif isinstance(arg, np.ndarray):
            if arg.ndim == 0:
                yield arg  # can't iterate over a 0-dim array
            else:
                yield from _expand_args(arg)
        elif isinstance(arg, Iterable):
            yield from _expand_args(arg)
        else:
            yield arg

def is_graph_object(*obj):
    # return 'theano' in sys.modules and any(isinstance(o, _gettheano().graph.basic.Variable)
    return 'theano' in sys.modules and any(isinstance(o, cf.GraphTypes)
                                 for o in _expand_args(obj))
is_theano_object = is_graph_object
def is_constant(*obj):
    # Both symbolic and shared objects return False
    return 'theano' not in sys.modules or builtins.all(
        isinstance(c, cf.ConstantTypes)
        for c in _expand_args(obj))
def is_pure_symbolic(*var):
    """
    Todo
    ----
    There seems to be some redundancy between ``is_pure_symbolic(x)``
    and ``not graph.is_computable(x)``.
    """
    # return 'theano' in sys.modules and any(isinstance(v, _gettheano().tensor.TensorVariable)
    return 'theano' in sys.modules and any(isinstance(v, cf.PureSymbolicTypes)
                                 for v in _expand_args(var))
is_theano_variable = is_pure_symbolic
def is_symbolic(*var):
    return 'theano' in sys.modules and builtins.any(
        isinstance(v, cf.GraphTypes)
        and not isinstance(v, cf.ConstantTypes)
        for v in _expand_args(var))
    issymbolic = is_symbolic  # With NumPy having no consistent convention, it's nigh to impossible to memorize, so just accept both
def is_shimmed_or_symbolic(*var):
    return any(isinstance(v, cf.ShimmedAndGraphTypes) for v in _expand_args(var))
def isshared(var):
    return isinstance(var, cf.SharedTypes)
# def isshared(*var):
#     if 'theano' in sys.modules:
#         return any(isinstance(v, (cf.SymbolicSharedType, ShimmedTensorShared))
#                    for v in _expand_args(var))
#     else:
#         return any(isinstance(v, ShimmedTensorShared)
#                    for v in _expand_args(var))

#######################
# Casting functions

def can_cast(from_, dtype, casting='safe'):
    # As far as I can tell, `np.can_cast` also works on Theano types.
    return np.can_cast(from_, dtype, casting=casting)

def cast(x, dtype, same_kind=True):
    """
    Parameters
    ----------
    x: scalar, array or Theano variable
        The variable to cast
    dtype: str
        The type to which cast the variable. One of
        - 'int8'
        - 'int16'
        - 'int32'
        - 'int64'
        - 'uint8'
        - 'uint16'
        - 'uint32'
        - 'uint64'
        - 'float16'
        - 'float32'
        - 'float64'
        Can optionally be an equivalent numpy dtype, as returned by
        <numpy var>.dtype.
    same_kind: bool
        When `same_kind` is `True`, only casts e.g. between 'float32' and 'float64'
        are permitted; others raise `TypeError`.

    """
    if isinstance(dtype, np.dtype):
        dtype = str(dtype)
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype = str(np.dtype(dtype))
    elif dtype == 'floatX':
        dtype = cf.floatX

    if same_kind:
        # Test that arguments are of the same kind
        # We get the 'kind' by stripping the number from dtype's string
        dtype_x = x.dtype if hasattr(x, 'dtype') else asarray(x).dtype
        kind_x = ''.join(c for c in str(dtype_x) if c.isalpha())
        kind_dtype = ''.join(c for c in str(dtype) if c.isalpha())
        if kind_x != kind_dtype:
            raise TypeError("Unsafe cast: trying to convert a {} to a {}. "
                            "If you want to disable this check, pass "
                            "`same_kind=False` to `cast()`"
                            .format(asarray(x).dtype, dtype))
    # FIXME: Why did I need this test ? If I have a plain Python variable,
    #        I *should* cast to a numpy dtype.
    # if str(asarray(x).dtype) == dtype:
    #     # Already the right dtype; no conversion to make
    #     return x
    if is_theano_object(x):
        return T.cast(x, dtype)
    elif hasattr(x, 'astype'):
        return x.astype(dtype)
    else:
        if np.__version__ >= '1.19':
            return np.dtype(dtype).type(x)
        else:
            # I don't remember which NumPy version I was using when I added the
            # keepdims arg, but I'm pretty sure it was required then
            return np.dtype(dtype).type(x, keepdims=True)

def cast_floatX(x, same_kind=True):
    return cast(x, dtype=cf.floatX, same_kind=same_kind)

#####################
# Simple convenience functions
def round(x):
    try:
        res = x.round()  # Theano variables have a round method
    except AttributeError:
        res = round(x)
    return res

def asvariable(x, dtype=None, name=None):
    if 'theano' in sys.modules:
        # No `isinstance` here: the point is to cast to a Theano variable
        if dtype is not None:
            return cast(T.as_tensor_variable(x, name=name), dtype)
        else:
            return T.as_tensor_variable(x, name=name)
    else:
        return np.asarray(x, dtype=dtype)

def asarray(x, dtype=None, broadcastable=None, symbolic=None):
    """Make x array-like.
    Note that if broadcastable is not None, and that Theano is loaded,
    the return value will always be a Theano variable, even if x is
    pure Python or Numpy. This is because `broadcastable` is a Theano-only
    property.

    Parameters
    ----------
    x: (scalar | array) | (numeric | symbolic)
        The value we want to ensure is array-like.
    dtype: str | dtype  (optional)
        If ≠ None, ensure the result is of this type
    broadcastable: Tuple[bool]  (optional)
        Broadcast pattern
        This is a Theano-only argument, and will force the result to be symbolic.
    symbolic: bool
        Override automatic selection of numeric vs symbolic. Useful to force
        symbolic output when the inputs are all numeric.
        Setting ``symbolic=False`` with symbolic arguments or `broadcastable`
        ≠ ``None`` will raise an error.

    Raises
    ------
    ValueError:
        If `x` is symbolic or `broadcastable` ≠ ``None``, but `symbolic` is
        ``False``.
    TypeError:
        If `x` is symbolic or `broadcastable` ≠ ``None`` but
        `config.use_theano` is False.
    """

    _symbolic = 'theano' in sys.modules and isinstance(x, _gettheano().graph.basic.Variable)
    if symbolic is None:
        symbolic = _symbolic
    elif symbolic is False and _symbolic is True:
        raise ValueError("Cannot force a symbolic variable to be numeric.")
    if (symbolic or broadcastable is not None) and not cf.use_theano:
        raise TypeError("Attempting to create a symbolic array while "
                        "`shim.config.use_theano` is False.")

    if symbolic:
        T = _getT()
        if dtype is not None:
            retval = T.cast(T.as_tensor_variable(x), dtype)
        else:
            retval = T.as_tensor_variable(x)
    else:
        retval = np.asarray(x, dtype=dtype)
        if cf.use_theano and broadcastable is not None:
            # Only Theano variables carry broadcasting information
            retval = T.as_tensor_variable(retval)

    if broadcastable is not None:
        for i, (vc, vn) in enumerate(zip(retval.broadcastable,
                                            broadcastable)):
            if vc != vn and vn:
                retval = T.addbroadcast(retval, i)
            elif vc != vn and not vn:
                retval = T.unbroadcast(retval, i)
    return retval

def asscalar(x):
    if isscalar(x) and not hasattr(x, 'ndim'):
        # shim.isscalar(x) returns True for 0-dim arrays
        return x
    elif is_theano_object(x):
        if all(x.broadcastable):
            return T.flatten(x)[0]
        else:
            raise ValueError("To cast a Theano tensor as a scalar, "
                             "all its dimensions must be broadcastable.")
    else:
        return np.asscalar(x)

def isscalar(x):
    """
    Return True if `x` is a scalar.
    Note that in contrast to Numpy's isscalar, this returns True for 0-dim arrays.
    """
    arrayed_x = asarray(x)
    return asarray(x).ndim == 0 and arrayed_x.dtype != 'object'

def isarray(x):
    # Some scalar numpy types (e.g. np.int64) have the 'ndim' attribute
    return (not np.isscalar(x)) and hasattr(x, 'ndim')

def issparse(var):
    """Return True if `var` is any recognized sparse format."""
    if 'theano.sparse' in sys.modules:
        return (sp.sparse.issparse(var)
                or isinstance(var, sys.modules['theano.sparse'].basic.SparseVariable))
    else:
        return sp.sparse.issparse(var)
def isspsparse(var):
    """Return True if `var` is sparse with `scipy.sparse` interface.
    True for scipy.sparse, theano.sparse."""
    if 'theano.sparse' in sys.modules:
        return (sp.sparse.issparse(var)
                or isinstance(var, sys.modules['theano.sparse'].basic.SparseVariable))
    else:
        return sp.sparse.issparse(var)

def flatten(x, outdim=1):
    if 'theano' in sys.modules and isinstance(x, theano.graph.basic.Variable):
        return T.flatten(x, outdim)
    else:
        outshape = x.shape[:outdim-1] + (np.prod(x.shape[outdim-1:]), )
        return x.reshape(outshape)

def addbroadcast(x, *axes):
    """
    Equivalent to theano.tensor.addbroadcast.
    For NumPy objects, checks that broadcasted dimensions have length 1,
    but otherwise does nothing.
    Compared to the Theano version, negative values for axes are supported:
    -1 refers to the last axis, -2 to the second last, etc.
    """
    if is_theano_object(x):
        # T.addbroadcast only works with positive axes
        axes = [ ax if ax >= 0 else x.ndim + ax for ax in axes ]
        return T.addbroadcast(x, *axes)
    else:
        for ax in axes:
            if x.shape[ax] != 1:
                raise ValueError("Tried to make axis {} of a variable with shape {} broadcastable. "
                                 "Only dimensions with length 1 can be broadcasted."
                                 .format(ax, x.shape))
        return x

# def eval(x, *args, **kwargs):
#     """
#     If `x` is has an 'eval' method, return `x.eval(*args, **kwargs)`. Otherwise just
#     return `x`. In the latter case, `*args` and `**kwargs` are ignored, and a
#     warning is printed if they are not empty.
#     """
#     if hasattr(x, 'eval'):
#         return x.eval(*args, **kwargs)
#     else:
#         if len(args) + len(kwargs) > 0:
#             logger.warning("Ignoring arguments to `eval`: object does not have "
#                            "an `eval` method.")
#         return x

#####################
# Convenience function for max / min

def largest(*args):
    """Element-wise max operation."""
    assert(len(args) >= 0)
    if len(args) == 1:
        return args[0]
    if 'theano' in sys.modules and any(isinstance(arg, _gettheano().graph.basic.Variable) for arg in args):
        return _getT().largest(*args)
    else:
        retval = np.maximum(args[0], args[1])
        for arg in args[2:]:
            retval = np.maximum(retval, arg)
        return retval

def smallest(*args):
    """Element-wise min operation."""
    assert(len(args) > 0)
    if len(args) == 0:
        return args[0]
    if 'theano' in sys.modules and any(isinstance(arg, _gettheano().graph.basic.Variable) for arg in args):
        return _getT().smallest(*args)
    else:
        retval = np.minimum(args[0], args[1])
        for arg in args[2:]:
            retval = np.minimum(retval, arg)
        return retval

def abs(x):
    if 'theano' in sys.modules and isinstance(x, _gettheano().graph.basic.Variable):
        if x.ndim == 2:
            return __builtins__['abs'](x)
        else:
            # Theano requires 2D objects for abs
            shape = x.shape
            return __builtins__['abs'](add_axes(x.flatten())).reshape(shape)
    else:
        return __builtins__['abs'](x)

######################
# Logical and comparison operators

def lt(a, b):
    if (cf.use_theano and (isinstance(a, theano.graph.basic.Variable)
                        or isinstance(b, theano.graph.basic.Variable))):
        return T.lt(a, b)
    else:
        return a < b
def le(a, b):
    if (cf.use_theano and (isinstance(a, theano.graph.basic.Variable)
                        or isinstance(b, theano.graph.basic.Variable))):
        return T.le(a, b)
    else:
        return a <= b
def gt(a, b):
    if (cf.use_theano and (isinstance(a, theano.graph.basic.Variable)
                        or isinstance(b, theano.graph.basic.Variable))):
        return T.gt(a, b)
    else:
        return a > b
def ge(a, b):
    if (cf.use_theano and (isinstance(a, theano.graph.basic.Variable)
                        or isinstance(b, theano.graph.basic.Variable))):
        return T.ge(a, b)
    else:
        return a >= b
def eq(a, b):
    if (cf.use_theano and (isinstance(a, theano.graph.basic.Variable)
                        or isinstance(b, theano.graph.basic.Variable))):
        return T.eq(a, b)
    else:
        return a == b

def bool(a):
    """
    Call this function on any expression that might
    appear in a Theano graph as a boolean (Theano expects
    integers rather than booleans.)
    """
    # Booleans need to be converted to integers for Theano
    if cf.use_theano and isinstance(a, (builtins.bool, np.bool_)):
        return np.int8(a)
    elif cf.use_theano or is_theano_object(a):
        return a
    else:
        return builtins.bool(a)

def and_(a, b):
    # Special case scalars so they don't return length 1 arrays
    if isscalar(a) and isscalar(b):
        return bool(bool(a) * bool(b))

    # matrix function
    if ('theano' in sys.modules and (isinstance(a, _gettheano().graph.basic.Variable)
                                or isinstance(b, _gettheano().graph.basic.Variable))):
        return _getT().and_(a, b)
    else:
        return np.logical_and(a, b)

def or_(a, b):
    # Special case scalars so they don't return length 1 arrays
    if isscalar(a) and isscalar(b):
        return bool(bool(a) + bool(b))

    # matrix function
    if ('theano' in sys.modules and (isinstance(a, _gettheano().graph.basic.Variable)
                                or isinstance(b, _gettheano().graph.basic.Variable))):
        return _getT().or_(a, b)
    else:
        return np.logical_or(a, b)

######################
# Conditionals

def ifelse(condition, then_branch, else_branch, name=None, outshape=None):
    """
    All parameters except `outshape` are the same as for theano.ifelse.ifelse

    `outshape` is an extra parameter to allow the then_branch and else_branch
    to have a different shape: the output will be reshaped into this form, but
    only if Theano is used. The reason we need this is as follows:
    Suppose we have a vector x which should be reshaped to (2,2). We might write
    (in pseudocode)
    ifelse(x.shape == (2,),
           concatenate((x, x)),
           x.reshape((2,2)))
    The Python version of this code has no trouble, because the correct branch
    will always reshape to (2,2). However, the Theano version wants a result with
    a well defined shape. Here the branch with `concatenate((x,x))` won't in
    general have the same shape as `x.reshape((2,2))`.
    We can get around this by defining outshape=(2,2) and writing instead
    ifelse(x.shape == (2,),
           concatenate((x, x)).reshape(outshape),
           x.reshape((2,2)).reshape(outshape))
    Now this makes Theano happy, but Python with its greedy evaluation
    evaluates both arguments before calling ifelse. So if x.shape=(2,2), the
    call will fail on `concatenate((x,x)).reshape(outshape)`. The solution
    is to only apply the reshape when using Theano, which is what specifying
    `outshape` as an argument does.
    """
    # First check if we can replace an Theano conditional by a Python one
    if is_theano_object(condition) and is_constant(condition):
        condition = bool(condition.data)

    # Now the actual function
    if (cf.use_theano
        and not isinstance(condition, builtins.bool)
        and (isinstance(condition, theano.graph.basic.Variable)
             or isinstance(then_branch, theano.graph.basic.Variable)
             or isinstance(else_branch, theano.graph.basic.Variable))):
        # Theano function
        if isinstance(then_branch, LazyEval):
            then_branch = then_branch.eval()
        if isinstance(else_branch, LazyEval):
            else_branch = else_branch.eval()
        if outshape is None:
            # We call `bool` on the condition, in case it's a Python boolean
            # (even shim.ge & friends can return bools)
            return theano.ifelse.ifelse(bool(condition), then_branch,
                                        else_branch, name)
        else:
            return theano.ifelse.ifelse(bool(condition), then_branch.reshape(outshape),
                                        else_branch.reshape(outshape), name)
    else:
        # Python function
        if condition:
            if isinstance(then_branch, LazyEval):
                then_branch = then_branch.eval()
            return then_branch
        else:
            if isinstance(else_branch, LazyEval):
                else_branch = else_branch.eval()
            return else_branch

def switch(cond, ift, iff):
    """
    For the equivalent to the single-argument version of `np.where`,
    see `nonzero`.
    """
    if (cf.use_theano and (isinstance(cond, theano.graph.basic.Variable)
                        or isinstance(ift, theano.graph.basic.Variable)
                        or isinstance(iff, theano.graph.basic.Variable))):
        return T.switch(cond, ift, iff)
    else:
        return np.where(cond, ift, iff)
where = switch
where.__doc__ = """Alias for `switch`."""

#####################
# Loop constructs

def scan(fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None,
         truncate_gradient=-1, go_backwards=False, mode=None, name=None, profile=False,
         allow_gc=None, strict=False, return_list=False):
    """
    WIP: Does not support taps. When using NumPy, every argument after `n_steps`
    except `return_list` is ignored.
    """
    if is_theano_object(sequences, outputs_info, non_sequences, n_steps):
        return gettheano().scan(
            fn, sequences, outputs_info, non_sequences, n_steps,
            truncate_gradient, go_backwards, mode, name, profile, allow_gc,
            strict, return_list)
    else:
        if not isinstance(sequences, (tuple, list)):
            sequences = (sequences,)
        if non_sequences is None:
            non_sequences = ()
        if isinstance(outputs_info, dict):
            raise TypeError("Taps not yet supported.")
        if n_steps is None:
            n_steps = len(sequences[0])

        accumulator = [np.zeros((n_steps,) + o.shape) for o in outputs_info]
        cur_val = outputs_info
        for t, i in zip(zip(*sequences), range(n_steps)):
            cur_val, updates = fn(*t, *cur_val, *non_sequences)
            for a, v in zip(accumulator, cur_val):
                a[i] = v

        if len(accumulator) == 1 and not return_list:
            accumulator = accumulator[0]
        if len(updates) > 0:
            logger.warning("NumPy `scan` produced updates for: {}, which were "
                           "ignored.".format(updates.keys()))
        return accumulator, updates

#####################
# Random number generation

class NumpyRNG(np.random.RandomState):
    """
    Note: For compatibility with Theano random streams, `size=None` is
          replaced with `size=()`, which returns a scalar array instead of
          a plain float.
    """
    # We inherit from the legacy RNG because that's what Theano uses.
    # def __init__(self, seed=None):
    #     self.seed(seed)
    #
    # def seed(self, seed=None):
    #     np.random.seed(seed)
    #
    def normal(self, size=(), avg=0.0, std=1.0, ndim=None, name=None):
        return super().normal(loc=avg, scale=std, size=size)

    def uniform(self, size=(), low=0.0, high=1.0, ndim=None, name=None):
        return super().uniform(low, high, size)

    def binomial(self, size=(), n=1, p=0.5, ndim=None, name=None):
        return super().binomial(n, p, size)

    @property
    def gen_seedgen(self):
        return self

def make_TheanoRNG(rng_class):
    """
    This function is deprecated if you can import `RandomStream` from
    `theano.tensor.random.utils`.
    """
    def add_kwarg_name(f):
        def wrapper(self, *args, **kwargs):
            name = kwargs.pop('name', None)
            sf = getattr(super(type(self), self), f.__name__)
            rndstream = sf(*args, **kwargs)
            if name is not None: rndstream.name = name
            return rndstream
        return wrapper

    class TheanoRNG(rng_class):
        """
        Wraps Theano RNG to allow for passing `name` as keyword argument when
        instantiating a random stream.
        """

        @add_kwarg_name
        def normal(self, size=(), avg=0.0, std=1.0, ndim=None, name=None):
            pass
        @add_kwarg_name
        def uniform(self, size=(), low=0.0, high=1.0, ndim=None, name=None):
            pass
        @add_kwarg_name
        def binomial(self, size=(), n=1, p=0.5, ndim=None, name=None):
            pass
    return TheanoRNG

def copy_random_state(from_rng, to_rng):
    """
    Set the state of the random number generator (RNG) :param:to so that it
    matches that of :param:from.

    Parameters
    ----------
    from:  theano RandomStream | MRG_RandomStream
    to: theano RandomStream | MRG_RandomStream
    """
    # Based on a function defined in the Theano docs: http://deeplearning.net/software/theano/tutorial/examples.html#copying-random-state-between-theano-graphs
    # Ensure the two RNGs are of the same type
    assert type(from_rng) is type(to_rng)
    # Ensure that their state updates are consistent
    # `str(su1[1])` retrieves something like `RandomFunction{uniform}.1`
    assert len(from_rng.state_updates) == len(to_rng.state_updates)
    assert all(str(su1[1]) == str(su2[1])
               for su1, su2 in zip(from_rng.state_updates,
                                   to_rng.state_updates))
    if isinstance(from_rng, _get_rng_mrg().MRG_RandomStream):
        to_rng.rstate = from_rng.rstate
    for (su1, su2) in zip(from_rng.state_updates, to_rng.state_updates):
        su2[0].set_value(su1[0].get_value())

def reseed_rng(rng, new_seed):
    """
    For Numpy legacy RandomState, just calls `rng.seed`.
    For Numpy Generator, sets the state of the underlying `BitGenerator` as
    though it had just been created with `BitGenerator(new_seed)`.
    For Theano, reseeds both the seeds of the current random streams, and
    the seed generator for future ones.
    """
    if isinstance(rng, np.random.RandomState):
        rng.seed(new_seed)
    elif isinstance(rng, np.random.Generator):
        rng.bit_generator.state = type(rng.bit_generator)(new_seed).state
    #elif is_symbolic(rng):
    elif isinstance(rng, cf.SymbolicNumpyRNGType):
        # I don't know why Theano chose to create a throwaway seedgen inside `seed`,
        # but it means that set reliable seeds for both current and new RNG streams,
        # we need to emulate `gen_seedgen` being used to reseed the RNGs.
        # `rng.seed` reseeds existing RNG streams, calling `seedgen.randint(2**30)`
        # as many times as there are RNG streams
        rng.seed(new_seed)
        # Reseed the gen_seedgen for new RNGs, and advance it as though it was
        # used in `seed`.
        rng.gen_seedgen.seed(new_seed)
        for i in range(len(rng.state_updates)):
            rng.randint(2**30)
    elif isinstance(rng, cf.SymbolicMRGRNGType):
        from .theano_types import MRG_RNG
        # Reset the rstate, and advance it as though it was
        # used in `seed`.
        rng.rstate = MRG_RNG(new_seed).rstate
        for i in range(len(rng.state_updates)):
            rng.randint(2**30)
    else:
        raise RuntimeError(f"Unrecognized RNG type; received {rng} (type: {type(rng)}).")

######################
# Tensor constructors

def shape_to_broadcast(shape):
    """
    Returns the default broadcastable pattern for a shape, replacing
    1s with `True`.
    """
    return tuple(n==1 for n in shape)

def tensor(object, name=None, dtype=None):
    """
    Make an object into a tensor. If `object` is a numpy array, a new tensor
    matching its shape and dtype is returned. The array values are used to set
    the test value.

    Not implemented: creating tensors from scalar objects.

    Examples:
    >>> import numpy as np
    >>> import theano_shim as shim
    >>> a = np.arange(5)
    >>> x = shim.tensor(a)
    >>> x2 = shim.tensor(a, name='a')
    >>> y = shim.tensor((5,), dtype='float64')
    >>> z = shim.tensor((5,3), name='z', dtype='int32')
    """
    # Try to infer the tensor shape, test_value, dtype and broadcast pattern
    broadcastable = None
    shape = None
    if isinstance(object, np.ndarray):
        # Numpy arrays become the symbolic's test value
        shape = object.shape
        test_value = object
        if dtype is None: dtype = object.dtype
        broadcastable = shape_to_broadcast(shape)
    elif isinstance(object, Number):
        # Scalar inputs become 0-dim arrays
        shape = ()
        test_value = object
        if dtype is None: dtype = str(np.dtype(type(object)))
        broadcastable = ()
    elif hasattr(object, 'broadcastable'):
        # Theano symbolics end up here
        # shape = object.shape   # This is going to be a symbolic expression
        if dtype is None: dtype = object.dtype
        broadcastable = object.broadcastable
        if name is None:
            name = f"{object.name} (tensor)"
        if hasattr(object.tag, 'test_value'):
            test_value = object.tag.test_value
        elif isshared(object):
            test_value = object.get_value()
        else:
            # Not possible to set test_value
            test_value = None
        if not cf.use_theano:
            raise TypeError("Somehow you specified what looks like a symbolic "
                            "object, yet Theano is not loaded.\n"
                            f"object: {object}\ntype: {type(object)}")
    elif isinstance(object, tuple):
        # All we have is a shape – we use array of ones as test_value
        shape = object
        if dtype is None:
            raise TypeError(
                "You must specify `dtype` if `object` does not provide one.")
        test_value = np.ones(shape, dtype=dtype)
        broadcastable = shape_to_broadcast(shape)
    else:
        raise TypeError("Unrecognized input type for `theano_shim.tensor`: "
                        f"{object} (type: {type(object)}.")
    if not cf.use_theano:
        # `test_value` should be defined at this point
        return np.array(test_value, dtype=dtype)
    else:
        if broadcastable is None: broadcastable = shape_to_broadcast(shape)
        tensor = getT().tensor(dtype, broadcastable, name=name)
        if test_value is not None:
            tensor.tag.test_value = test_value
        return tensor

######################
# Shared variable constructor

class ShimmedTensorShared(np.ndarray):
    # See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    # for indications on subclassing ndarray

    def __new__(cls, value, name=None, strict=False, allow_downcast=None, **kwargs):
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if hasattr(value, 'shim_class'):
            cls = value.shim_class
        obj = value.view(cls).copy()
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)

    # We are emulating theano.shared, where different instances
    # are considered distinct
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)

    def _as_TensorVariable(self):
        # Allow mixing of ShimmedTensorShared and Theano variables
        # Theano looks for this function first when multiplying with non-Theano types
        if cf.use_theano:
            return T.constant(self.get_value())
        else:
            return self.get_value()

    # Usual theano.shared interface
    def get_value(self, borrow=False, return_internal_type=False):
        return self.view(np.ndarray)
            # On values obtained by get_value, equality testing shold
            # follow the usual rules for arrays, hence the view(np.ndarray)

    def set_value(self, new_value, borrow=False):
        """
        If `allow_resize` is false (default), will raise an error if
        new_value has a different shape than the stored variable.
        """
        new_value = np.asarray(new_value)
        try:
            if self.shape != new_value.shape:
                self.resize(new_value.shape, refcheck=False)
                # refcheck is necessary to get this to work, but bypasses
                # the reference checks. Reference errors might occur if
                # a reference to this ShimmedTensorShared variable exists elsewhere,
                # and we try to access it after the resize. This is the kind
                # of thing you shouldn't do anyway with Theano variables.
            self[:] = new_value
        except IndexError:
            # Scalars will fail on the above
            assert(isscalar(new_value))
                # np.isscalar will fail on 0-dim arrays; isscalar works
            self = super(ShimmedTensorShared, self).__setitem__(None, new_value)

    def eval(self, inputs_to_values=None):
        return self.get_value()

    @property
    def broadcastable(self):
        """For Numpy arrays, an axis is broadcastable iff it has length one."""
        return tuple(s==1 for s in self.shape)

cf.add_terminating_types([ShimmedTensorShared])
cf._shared_types += (ShimmedTensorShared,)

def shared(value, name=None, strict=False, allow_downcast=None, symbolic=True,
           **kwargs):
    """
    In contrast to Theano's `shared()`, the broadcast pattern is set to be
    compatible with NumPy's behaviour; i.e., any axis in `value` with dimension
    1 is considered broadcastable by default.
    As with Theano's `shared()`, broadcast pattern can by changed by passing
    the :param:broadcastable keyword argument.
    """
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
    if 'dtype' in kwargs:
        logger.warning("You passed the keyword 'dtype' to the shared constructor. "
                       "Theano doesn't support this keyword for shared variables.")
    if symbolic and cf.use_theano:
        # Unless a broadcast pattern is specified, we create one to match
        # the NumPy behaviour (broadcastable on all axes of dimension 1).
        broadcast_pattern = kwargs.pop('broadcastable', None)
        if broadcast_pattern is None:
            broadcast_pattern = tuple(True if s==1 else False for s in value.shape)
        return theano.shared(value, name, strict, allow_downcast,
                             broadcastable=broadcast_pattern, **kwargs)
    else:
        return ShimmedTensorShared(value, name, strict, allow_downcast, **kwargs)

######################
# Interchangeable set_subtensor
def set_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if 'theano' in sys.modules and (isinstance(x, _gettheano().graph.basic.Variable)
                                    or isinstance(y, _gettheano().graph.basic.Variable)):
        return _getT().set_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert x.base is not None
            # Ensure that x is a view of another ndarray
        assert x.shape == y.shape
        x[:] = y
        return x.base

def inc_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if 'theano' in sys.modules and (isinstance(x, _gettheano().graph.basic.Variable)
                                    or isinstance(y, _gettheano().graph.basic.Variable)):
        return T.inc_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert x.base is not None
            # Ensure that x is a view of another ndarray
        # assert x.shape == y.shape
        x[:] += y
        return x.base

# TODO: Deprecate: numpy arrays have ndim
def get_ndims(x):
    if cf.use_theano and isinstance(x, theano.graph.basic.Variable):
        return x.ndim
    else:
        return len(x.shape)

######################
# Axis manipulation functions
# E.g. to treat a scalar as a 1x1 matrix

def reshape(array, newshape, ndim=None):
    if is_theano_object(array):
        return array.reshape(newshape, ndim)
    else:
        return array.reshape(newshape)

def atleast_1d(*arrays):
    """
    In contrast to `numpy.atleast_1d`, will not cast lists or tuples to arrays.
    This is to allow lists of symbolic variables.
    """
    if len(arrays) == 1:
        a = arrays[0]
        if isscalar(a):
            a = add_axes(a, 1)
        return a
    else:
        assert len(arrays) > 1
        return [atleast_1d(a) for a in arrays]

def add_axes(x, num=1, pos='left'):
    """
    Add an axis to `x`, e.g. to treat a scalar as a 1x1 matrix.
    String arguments for `pos` should cover most typical use cases;
    for more complex operations, like adding axes to the middle,
    specify the insertion position for the axes directly.

    Parameters
    ----------
    num: int
        Number of axes to add. Default: 1.
    pos: 'before' | 'left' | 'after' | 'right' | 'before last' | int
        - 'before', 'left', 'begin', 'last' turns a 1D vector into a row vector. (Default)
        - 'after', 'right', 'end', 'first' turns a 1D vector into a column vector.
        - 'before last' adds axes to the second-last position.
          Equivalent to 'left' on 1D vectors.'.
        - An integer adds the axes before this position
            + 0 : equivalent to 'before'
            + -1 : equivalent to 'before last'
            + `x.ndim` : equivalent to 'after'
    """
    if is_theano_object(x):
        if pos in ['left', 'before', 'begin', 'first']:
            shuffle_pattern = ['x']*num
            shuffle_pattern.extend(range(x.ndim))
        elif pos  in ['right', 'after', 'end', 'last']:
            shuffle_pattern = list(range(x.ndim))
            shuffle_pattern.extend( ['x']*num )
        elif pos == 'before last':
            shuffle_pattern = list(range(x.ndim))
            shuffle_pattern = shuffle_pattern[:-1] + ['x']*num + shuffle_pattern[-1:]
        else:
            try:
                shuffle_pattern = list(range(x.ndim))
                shuffle_pattern = shuffle_pattern[:pos] + ['x']*num + shuffle_pattern[pos:]
            except TypeError:
                raise ValueError("Unrecognized argument `{}` for pos.".format(pos))
        return x.dimshuffle(shuffle_pattern)
    else:
        x = np.asarray(x)
        if pos in ['left', 'before', 'begin', 'first']:
            return x.reshape( (1,)*num + x.shape )
        elif pos in ['right', 'after', 'end', 'last']:
            return x.reshape( x.shape + (1,)*num )
        elif pos == 'before last':
            return x.reshape( x.shape[:-1] + (1,)*num + x.shape[-1:] )
        else:
            try:
                return x.reshape( x.shape[:pos] + (1,)*num + x.shape[pos:] )
            except TypeError:
                raise ValueError("Unrecognized argument {} for pos.".format(pos))

def moveaxis(a, source, destination):
    if is_theano_object(x):
        axes_lst = list(range(x.ndim))
        axes_lst.pop(source)
        axes_lst = axes_lst[:destination] + [source] + axes_lst[destination:]
        return a.dimshuffle(axes_lst)
    else:
        return np.moveaxis(a, source, destination)

def pad(array, array_shape, pad_width, mode='constant', **kwargs):
    """
    All parameters except `array_shape` are the same as for np.pad.
    `array_shape` is necessary because while we can deal with a Theano array,
    we need to know its shape.
    """
    if mode not in ['constant']:
        raise ValueError("theano_shim does not support mode '{}'".format(mode))
    if not is_theano_object(array):
        assert(array.shape == array_shape)
            # If this fails, than the Theano code will also fail
            # (perhaps cryptically).
        return np.pad(array, pad_width, mode, **kwargs)
    elif isshared(array):
        assert(array.get_value(borrow=True).shape == array_shape)
        return np.pad(array.get_value(borrow=True), pad_width, mode, **kwargs)
    else:
        def expand_arg(arg):
            if isscalar(arg):
                arg = (arg, arg) # before, after
            if isscalar(arg[0]):
                if len(arg) == 1:
                    arg = (arg[0], arg[0])
                arg = (arg,)
            if len(arg) == 1:
                assert(isinstance(arg, (tuple, list)))
                arg = arg * array.ndim
            assert(len(arg) == array.ndim)
            assert(all(len(tup) == 2 for tup in arg))
            return arg
        pad_width = expand_arg(pad_width)
        if mode == 'constant':
            vals = kwargs.pop('constant_values', None)
            if vals is None:
                vals = 0
            vals = expand_arg(vals)

            res = array
            new_shape = tuple( w[0] + shape + w[1]
                               for w, shape in zip(pad_width, array_shape) )
            for i, (w, v) in enumerate(zip(pad_width, vals)):
                if (w[0] != 0 or w[1] != 0):
                    shape1 = new_shape[:i] + (w[0],) + array_shape[i+1:]
                    shape2 = new_shape[:i] + (w[1],) + array_shape[i+1:]
                    res = T.concatenate( ( np.ones(shape1)*v[0],
                                           res,
                                           np.ones(shape2)*v[1]),
                                         axis=i)

        return res


########################
# Functions from scipy.misc

def factorial(n, exact=False):
    """Note: the Theano version uses `gamma` regardless of `exact`"""
    assert(istype(n, 'int'))
    check(np.all(n >= 0))
    if is_theano_object(n):
        return T.gamma(n+1)
    else:
        return sp.misc.factorial(n, exact)


########################
# Wrapper for discrete 1D convolutions

# TODO: Use fftconvolve if ~500 time bins or more

def conv1d(data_arr, kernel_arr, tarr_len, discrete_kernel_shape, mode='valid'):
    """
    Convolve each component of data_arr with kernel_arr and stack the result
    into an array. data_arr is an NxM array, where N is the number of time bins
    and M the number of components kernel_arr is an MxM array, for which the
    element with index (i,j) represents the contribution of component j to
    component i. (Consistent with a dot product where the kernel is on the left.)
    In other words, each row j of kernel_arr is convolved with the row j of data_arr.

    Parameters
    ----------
    data_arr : 2D ndarray or theano.tensor
        NxM array
    kernel_arr : 2D ndarray | theano.tensor
        MxM array
    tarr_shape : tuple
        The length of the history's time array. Theano can't determine the
        shape from a tensor, so it is specified separately. When computing
        using NumPy, validated agains data_arr.shape[0]
    discrete_kernel_shape : tuple
        Shape of the discrete kernel array. Theano can't determine the shape
        from a tensor, so it is specified separately. When computing using
        NumPy, this is checked for consistency.

    Returns
    -------
    ndarray:
        Result has shape (M, M)

    """

    assert(data_arr.ndim == 2)
    output_shape = discrete_kernel_shape[1:]
    if (kernel_arr.ndim == 2):
        # Algorithm assumes a "to" axis on the kernel. Add it.
        kernel_arr = add_axes(kernel_arr, 1, 'before last')
        discrete_kernel_shape = discrete_kernel_shape[0:1] + (1,) + discrete_kernel_shape[1:2]
    else:
        check(kernel_arr.ndim == 3)

    # Convolutions leave the time component on the inside, but we want it on the outside
    # So we do the iterations in reverse order, and flip the result with transpose()
    # The result is indexed as [tidx][to idx][from idx]
    if cf.use_theano:
        # We use slices from_idx:from_idx+1 because conv2d expects 2D objects
        # We then index [:,0] to remove the spurious dimension
        result = T.stack(
                  [ T.stack(
                       [ T.signal.conv.conv2d(data_arr[:, from_idx:from_idx+1 ],
                                              kernel_arr[:, to_idx, from_idx:from_idx+1 ],
                                              image_shape = (tarr_len, 1),
                                              filter_shape = (discrete_kernel_shape[0], 1),
                                              border_mode = mode)[:,0]
                         for to_idx in np.arange(discrete_kernel_shape[1]) ] )
                       for from_idx in np.arange(discrete_kernel_shape[2]) ] ).T
    else:
        assert(discrete_kernel_shape == kernel_arr.shape)
        assert(tarr_len == data_arr.shape[0])
        result = np.stack(
                  [ np.stack(
                       [ scipy.signal.convolve(data_arr[:, from_idx ],
                                               kernel_arr[:, to_idx, from_idx ],
                                               mode=mode)
                         for to_idx in np.arange(kernel_arr.shape[1]) ] )
                       for from_idx in np.arange(kernel_arr.shape[2]) ] ).T

    return result.reshape((tarr_len - discrete_kernel_shape[0] + 1,) + output_shape)


def lfilter(size, b, a, x, *args, **kwargs):
    """
    Wrapper for the linear filter operator implemented by scipy.signal.lfilter

    At the moment, the implementation is restricted to the case a = 1.

    :param b: array of size M. The moving average coefficients.
    :param a: array of size N. The autoregressive coefficients.
    :param x: array.
    :param size: tuple (M, N)
    :return:
    """

    sym_a = is_theano_object(a)
    sym_b = is_theano_object(b)
    sym_x = is_theano_object(x)

    M, N = size
    if sym_b or sym_x:
        s = x * b[0]
        for tau in range(1, M):
            u = x[:-tau] * b[tau]
            s = T.inc_subtensor(s[tau:], u)
    else:
        s = scipy.signal.lfilter(b, a, x, *args, **kwargs)
    return s

################################
# Module initialization

load('numpy')
    # By default, don't load Theano

#####################
# Gradients

def grad(expr, wrt, *args, **kwargs):
    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]
    if not all(is_symbolic(w) for w in wrt):
        raise TypeError("Gradient must be with respect to symbolic variables.")
    if not is_symbolic(expr):
        raise TypeError("Expression must be symbolic.")
    # elif not set(wrt).issubset(shim.symbolic_inputs(expr)):
    #     raise TypeError("Attempted to take gradient with respect to the "
    #                     "following values, which are not part of the "
    #                     "computational graph: {}"
    #                     .format(', '.join(v.name for v in set(wrt).difference(
    #                         shim.symbolic_inputs(expr)))))
    return getT().grad(expr, wrt, *args, **kwargs)

#######################
# NumPy functions

def all(x):
    if is_theano_object(x):
        return T.all(x)
    else:
        return np.all(x)
def arange(start, stop=None, step=1, dtype=None, symbolic=None):
    _symb = is_theano_object(start, stop, step, dtype)
    if symbolic is None:
        symbolic = _symb
    elif _symb and not symbolic:
        raise TypeError("Attempting to create a symbolic array while "
                        "`shim.config.use_theano` is False.")
    if symbolic:
        dtype = str(np.dtype(dtype))  # Convert nptype and dtype to string
        return T.arange(start, stop, step, dtype)
    else:
        return np.arange(start, stop, step, dtype)
def broadcast_to(array, shape, subok=False):
    if is_theano_object(array, shape):
        return T.ones(shape) * array
    else:
        return np.broadcast_to(array, shape, subok)
def copy(array, symbolic=False, name=None):
    """
    NumPy `array`:
        Calls ``array.copy()``.
    Symbolic `array` & `symbolic` == True:
        Make a symbolic copy of the `array` by calling ``array.copy(name=name)``.
        `array` appears in the computational graph of the copy.
    Symbolic `array` & `symbolic` == False:
        Make a copy of the Python object by calling ``copy.copy(array)``.
        ``array`` does not appear in the computational graph of the copy, but
        inputs to ``array`` do.

    If `name` != `None` and `array` is symbolic, it is renamed accordingly.

    >>> import theano_shim as shim
    >>> shim.load('theano')
    >>> y = shim.tensor(np.array(3.), 'y')
    >>> z = shim.copy(y)
    >>> z2 = shim.copy(y**2)
    >>> zsymb = shim.copy(y, symbolic=True)
    >>> y in shim.graph.inputs(z)      # False
    >>> y in shim.graph.inputs(z2)     # True
    >>> y in shim.graph.inputs(zsymb)  # True
    """
    if not is_theano_object(array):
        return array.copy()
    elif symbolic:
        return array.copy(name=name)
    else:
        c = copymodule.copy(array)
        if name is not None:
            c.name = name
        return c
def concatenate(tensor_list, axis=0):
    if any(is_theano_object(x) for x in tensor_list):
        return T.concatenate(tensor_list, axis)
    else:
        return np.concatenate(tensor_list, axis)
def cos(x):
    if is_theano_object(x):
        return T.cos(x)
    else:
        return np.cos(x)
def cosh(x):
    if is_theano_object(x):
        return T.cosh(x)
    else:
        return np.cosh(x)
def clip(a, a_min, a_max):
    if is_theano_object(a, a_min, a_max):
        return T.clip(a, a_min, a_max)
    else:
        return np.clip(a, a_min, a_max)
def cumsum(x, axis=None, dtype=None):
    if is_theano_object(x):
        return T.cumsum(x, axis)
    else:
        return np.cumsum(x, axis, dtype)
def diag(x, k=0):
    if is_theano_object(x, k):
        return T.diag(x, k=k)
    else:
        return np.diag(x, k=k)
def dot(x, y):
    if is_theano_object(x) or is_theano_object(y):
        return T.dot(x, y)
    else:
        return np.dot(x, y)
def exp(x):
    if is_theano_object(x):
        return T.exp(x)
    else:
        # if isinstance(x, ShimmedTensorShared):
        #     x = x.get_value()
        return np.exp(x)
def gammaln(x):
    if is_theano_object(x):
        return T.gammaln(x)
    else:
        return sp.special.gammaln(x)
def isfinite(x, *args, **kwargs):
    """Always returns `True` on symbolic inputs."""
    if is_theano_object(x):
        return True
    else:
        assert not is_theano_object(kwargs.values())
        return np.isfinite(x, **kwargs)
def log(x):
    if is_theano_object(x):
        return T.log(x)
    else:
        return np.log(x)
def log10(x):
    if is_theano_object(x):
        return T.log10(x)
    else:
        return np.log10(x)
def max(x):
    if is_theano_object(x):
        return T.max(x)
    else:
        return np.max(x)
def mean(x):
    if is_theano_object(x):
        return T.mean(x)
    else:
        return np.mean(x)
def min(x):
    if is_theano_object(x):
        return T.min(x)
    else:
        return np.min(x)
def multiply(x, y):
    if is_theano_object(x, y):
        return x*y
    else:
        return np.multiply(x, y)
def nonzero(x):
    """
    Returns:
        (numeric  x) tuple of Array[int], one array per dimension.
        (symbolic x) Python tuple of symbolic Subtensor, one Subtensor per dimension.
    """
    if isscalar(x):
        raise ValueError("Nonzero only supports non-scalar arrays")
    if is_theano_object(x):
        return T.nonzero(x)
    else:
        return np.nonzero(x)
def ones(shape, dtype=None):
    if is_theano_object(shape):
        return T.ones(shape, dtype)
    else:
        return np.ones(shape, dtype)
def prod(x, *args):
    if is_theano_object(x):
        return T.prod(x, *args)
    else:
        return np.prod(x, *args)
def sin(x):
    if is_theano_object(x):
        return T.sin(x)
    else:
        return np.sin(x)
def sinh(x):
    if is_theano_object(x):
        return T.sinh(x)
    else:
        return np.sinh(x)
def sqrt(x):
    if is_theano_object(x):
        return T.sqrt(x)
    else:
        return np.sqrt(x)
def stack(tensors, axis=0):
    if is_theano_object(*tensors):
        return T.stack(tensors, axis)
    else:
        return np.stack(tensors, axis)
def sum(x, axis=None, dtype=None, acc_dtype=None, keepdims=np._NoValue):
    if is_theano_object(x):
        result = T.sum(x, axis, dtype, acc_dtype)
        if keepdims and keepdims is not np._NoValue:
            if not isinstance(axis, Iterable):
                axes = [axis]
            else:
                axes = sorted(axis)
            for axis in axes:
                result = add_axes(result, pos=axis)
        return result
    else:
        return np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
def tan(x):
    if is_theano_object(x):
        return T.tan(x)
    else:
        return np.tan(x)
def tanh(x):
    if is_theano_object(x):
        return T.tanh(x)
    else:
        return np.tanh(x)
def tile(x, reps, ndim=None):
    if is_theano_object(x):
        return T.tile(x, reps, ndim)
    else:
        return np.tile(x, reps)
def zeros(shape, dtype=None):
    if is_theano_object(shape):
        return T.zeros(shape, dtype)
    else:
        return np.zeros(shape, dtype)

def zeros_like(x, dtype=None):
    if is_theano_object(x):
        return T.zeros_like(x, dtype)
    else:
        return np.zeros_like(x, dtype)
