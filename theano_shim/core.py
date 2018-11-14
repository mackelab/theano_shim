"""
A simple convenient exchangeable interface, so we don't need
conditionals just to select between e.g. T.sum and np.sum.
More specific calls can be dealt with in the related code by
conditioning on this module's `use_theano` flag

This module provides an interchangeable interface to common operations,
such as type casting and checking, assertions and rounding, as well
as 'shim' datatypes for random number streams and shared variables.

Usage
-----
At the top of your code, include the line
`import theano_shim as shim`
By default this will not even try to load Theano, so you can use it on
a machine where Theano is not installed.
To 'switch on' Theano, add the following below the import:
`shim.load_theano()`
You can switch it back to its default state with `shim.load(False)`.


Pointers for writing theano switches
------------------------------------
- Type checking
    + isinstance(x, theano.tensor.TensorVariable) will be True when
      x is a theano variable, but False for wrappers around Python
      objects such as shared variables.
    + isinstance(x, theano.gof.Variable) is more inclusive, returning
      True for shared variables as well.
    + These two tests are provided by the `is_theano_variable` and
      `is_theano_object` convenience methods.
"""

import logging
import builtins
import collections
import inspect
from numbers import Number
import sys
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
    load(True)

def load(load_theano=True, reraise=False):
    """Reset the module to use or not use Theano.
    This should be called once at the top of your code.

    Parameters
    ----------
    use_theano: Boolean
        - True  : Module will act as an interface to Theano
        - False : Module will simulate Theano using pure Numpy
    reraise: Boolean
        If true, import errors will be reraised to allow them to propagate to the parent.
    """
    #TODO: Move as much as possible to config.Config
    #      And/or move the function to config, and import config.load

    global theano, T

    if load_theano:
        try:
            import theano
        except ImportError:
            logger.error("The theano library was not found.")
            cf.use_theano = False
            if reraise:
                raise
        else:
            cf.use_theano = True
    else:
        cf.use_theano = False

    if cf.floatX == 'float32':
        config.make_constants_32bit()

    if cf.use_theano:
        import theano.ifelse
        import theano.tensor as T
        cf.lib = T
        import theano.tensor.signal.conv
        import theano.sparse
        import theano.tensor.shared_randomstreams  # CPU only
        #from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU & GPU

        cf.inf = 1e12
        cf.RandomStreams = theano.tensor.shared_randomstreams.RandomStreams

        # if cf.sys.version_info.minor >= 5:
        #     cf.Numeric = cf.Union[np.ndarray, T.TensorVariable]

    else:
        cf.lib = np
        cf.inf = np.inf
        cf.RandomStreams = ShimmedRandomStreams

        # if cf.sys.version_info.minor >= 5:
        #     cf.Numeric = cf.Union[np.ndarray]

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
    logger.warning("Deprecation warning: theano_shim.is_computable() is deprecated. "
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
    """
    if isinstance(variable, dict):
        for key, val in variable.items():
            add_update(key, val)
    elif isinstance(variable, collections.Sequence):
        for key, val in variable:
            add_update(key, val)
    else:
        logger.debug("Adding Theano update : {} -> {}".format(variable.name, str(value)))
        if not isshared(variable):
            raise ValueError("The updates mechanism only applies to shared variables.")
        cf.theano_updates[variable] = value
add_updates = add_update

def get_updates():
    return cf.theano_updates

def reset_updates():
    logger.debug("Clearing Theano updates")
    cf.theano_updates = collections.OrderedDict()

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

def print(x, message=None, printfn='print', message_prefix="DEBUG - ",
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
        different types of outputs. Defaults to "DEBUG - ".
    file: file handle
        Where to print the value; default is 'sys.stdout'.
        Same argument as used in print() or theano.printing.debugprint.
    """
    if message is None:
        message = getattr(x, 'name', "")
        if message is None: message = ""  # x.name might be None
    if is_theano_object(x):
        msg = message_prefix + message
        if printfn == 'print':
            return theano.printing.Print(msg, global_fn=_get_print_fn(file))(x)
        elif printfn == 'debugprint':
            print(msg)
            theano.printing.debugprint(x, file=file)
            return x
        elif printfn == 'eval':
            try:
                val = x.eval()
            except theano.gof.fg.MissingInputError:
                return theano.printing.Print(msg, global_fn=_get_print_fn(file))(x)
            else:
                print(msg + " Value of {}: {}".format(str(x), val))
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

def pprint(x):
    """
    Call pretty printer (`pprint`) on Theano objects, otherwise standard `print`
    """
    if is_theano_object(x):
        return theano.printing.pprint(x)
    else:
        return str(x)

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
    if not cf.use_theano or not isinstance(stmt, theano.gof.Variable):
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
    elif 'theano' in sys.modules and isinstance(var, _gettheano().gof.Variable):
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
    if isinstance(type_str, str) or not isinstance(type_str, collections.Iterable):
        type_str = [type_str]
    # Ensure we only have strings (not dtypes)
    type_str = [str(ts) for ts in type_str]
    # Check type
    if ('theano' not in sys.modules
        or not isinstance(obj, _gettheano().gof.Variable)):
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
    if not isinstance(arglst, collections.Iterable):
        arglst = [arglst]
    elif ('theano' in sys.modules
          and isinstance(arglst, _gettheano().gof.Variable)):
        arglst = [arglst]
    elif isinstance(arglst, cf._TerminatingTypes):
        arglst = [arglst]
    for arg in arglst:
        if 'theano' in sys.modules and isinstance(arg, _gettheano().gof.Variable):
            # Theano variables aren't iterable
            yield arg
        elif isinstance(arg, cf._TerminatingTypes):
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
        elif isinstance(arg, collections.Iterable):
            yield from _expand_args(arg)
        else:
            yield arg

def is_theano_variable(*var):
    return 'theano' in sys.modules and any(isinstance(v, _gettheano().tensor.TensorVariable)
                                 for v in _expand_args(var))
def is_theano_object(*obj):
    return 'theano' in sys.modules and any(isinstance(o, _gettheano().gof.Variable)
                                 for o in _expand_args(obj))
def is_constant(*obj):
    # Both symbolic and shared objects return False
    return 'theano' not in sys.modules or builtins.all(
        isinstance(c, (_gettheano().tensor.TensorConstant, Number))
        for c in _expand_args(obj))
def isshared(*var):
    if 'theano' in sys.modules:
        return any(isinstance(v, (_getT().sharedvar.SharedVariable, ShimmedShared))
                   for v in _expand_args(var))
    else:
        return any(isinstance(v, ShimmedShared)
                   for v in _expand_args(var))

#######################
# Casting functions
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
    elif inspect.isclass(dtype) and issubclass(dtype, np.generic):
        dtype = str(dtype(1).dtype)  # FIXME Can we make this less ugly ?

    if same_kind:
        # Test that arguments are of the same kind
        # We get kind by stripping number from dtype's string
        kind_x = ''.join(c for c in str(asarray(x).dtype) if c.isalpha())
        kind_dtype = ''.join(c for c in str(dtype) if c.isalpha())
        if kind_x != kind_dtype:
            raise TypeError("Unsafe cast: trying to convert a {} to a {}. "
                            "If you want to disable this check, pass "
                            "`same_kind=False` to `cast()`"
                            .format(asarray(x).dtype, dtype))
    if str(asarray(x).dtype) == dtype:
        # Already the right dtype; no conversion to make
        return x
    elif is_theano_object(x):
        return T.cast(x, dtype)
    elif hasattr(x, 'astype'):
        return x.astype(dtype)
    else:
        val = ( np.int8(x, keepdims=True) if dtype == 'int8'
                else np.int16(x, keepdims=True) if dtype == 'int16'
                else np.int32(x, keepdims=True) if dtype == 'int32'
                else np.int64(x, keepdims=True) if dtype == 'int64'
                else np.uint8(x, keepdims=True) if dtype == 'uint8'
                else np.uint16(x, keepdims=True) if dtype == 'uint16'
                else np.uint32(x, keepdims=True) if dtype == 'uint32'
                else np.uint64(x, keepdims=True) if dtype == 'uint64'
                else np.float16(x, keepdims=True) if dtype == 'float16'
                else np.float32(x, keepdims=True) if dtype == 'float32'
                else np.float64(x, keepdims=True) if dtype == 'float64'
                else None )
        if val is None:
            raise ValueError("Unrecognized type {}.".format(dtype))
        return val

# def cast_int8(x):
#     if is_theano_object(x):
#         return T.cast(x, 'int8')
#     else:
#         return np.int8(x)
# def cast_int16(x):
#     if is_theano_object(x):
#         return T.cast(x, 'int16')
#     else:
#         return np.int16(x)
# def cast_int32(x):
#     if is_theano_object(x):
#         return T.cast(x, 'int32')
#     else:
#         return np.int32(x)
# def cast_int64(x):
#     if is_theano_object(x):
#         return T.cast(x, 'int64')
#     else:
#         return np.int64(x)

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

def asvariable(x, dtype=None):
    if cf.use_theano in sys.modules:
        # No `isinstance` here: the point is to cast to a Theano variable
        if dtype is not None:
            return T.cast(T.as_tensor_variable(x), dtype)
        else:
            return T.as_tensor_variable(x)
    else:
        return np.asarray(x, dtype=dtype)

def asarray(x, dtype=None, broadcastable=None):
    """Make x array-like.
    Note that if broadcastable is not none, and that Theano is loaded,
    the return value will always be a Theano variable, even if x is
    pure Python or Numpy. This is because `broadcastable` is a Theano-only
    property.
    """

    if 'theano' in sys.modules and isinstance(x, _gettheano().gof.Variable):
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
    if isscalar(x):
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

def flatten(x, outdim=1):
    if 'theano' in sys.modules and isinstance(x, theano.gof.Variable):
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

def eval(x, *args, **kwargs):
    """
    If `x` is has an 'eval' method, return `x.eval(*args, **kwargs)`. Otherwise just
    return `x`. In the latter case, `*args` and `**kwargs` are ignored, and a
    warning is printed if they are not empty.
    """
    if hasattr(x, 'eval'):
        return x.eval(*args, **kwargs)
    else:
        if len(args) + len(kwargs) > 0:
            logger.warning("Ignoring arguments to `eval`: object does not have "
                           "an `eval` method.")
        return x

#####################
# Convenience function for max / min

def largest(*args):
    """Element-wise max operation."""
    assert(len(args) >= 0)
    if len(args) == 1:
        return args[0]
    if 'theano' in sys.modules and any(isinstance(arg, _gettheano().gof.Variable) for arg in args):
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
    if 'theano' in sys.modules and any(isinstance(arg, _gettheano().gof.Variable) for arg in args):
        return _getT().smallest(*args)
    else:
        retval = np.minimum(args[0], args[1])
        for arg in args[2:]:
            retval = np.minimum(retval, arg)
        return retval

def abs(x):
    if 'theano' in sys.modules and isinstance(x, _gettheano().gof.Variable):
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
    if (cf.use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.lt(a, b)
    else:
        return a < b
def le(a, b):
    if (cf.use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.le(a, b)
    else:
        return a <= b
def gt(a, b):
    if (cf.use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.gt(a, b)
    else:
        return a > b
def ge(a, b):
    if (cf.use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.ge(a, b)
    else:
        return a >= b
def eq(a, b):
    if (cf.use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
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
    if ('theano' in sys.modules and (isinstance(a, _gettheano().gof.Variable)
                                or isinstance(b, _gettheano().gof.Variable))):
        return _getT().and_(a, b)
    else:
        return np.logical_and(a, b)

def or_(a, b):
    # Special case scalars so they don't return length 1 arrays
    if isscalar(a) and isscalar(b):
        return bool(bool(a) + bool(b))

    # matrix function
    if ('theano' in sys.modules and (isinstance(a, _gettheano().gof.Variable)
                                or isinstance(b, _gettheano().gof.Variable))):
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
        and (isinstance(condition, theano.gof.Variable)
             or isinstance(then_branch, theano.gof.Variable)
             or isinstance(else_branch, theano.gof.Variable))):
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
    if (cf.use_theano and (isinstance(cond, theano.gof.Variable)
                        or isinstance(ift, theano.gof.Variable)
                        or isinstance(iff, theano.gof.Variable))):
        return T.switch(cond, ift, iff)
    else:
        return np.where(cond, ift, iff)

#####################
# Shimmed random functions

class ShimmedRandomStreams:
    def __init__(self, seed=None):
        np.random.seed(seed)

    def normal(self, size=(), avg=0.0, std=1.0, ndim=None):
        return np.random.normal(loc=avg, scale=std, size=size)

    def uniform(self, size=(), low=0.0, high=1.0, ndim=None):
        return np.random.uniform(low, high, size)

    def binomial(self, size=(), n=1, p=0.5, ndim=None):
        return np.random.binomial(n, p, size)

######################
# Tensor constructors

def shape_to_broadcast(shape):
    """
    Returns the default broadcastable pattern for a shape, replacing
    1s with `True`.
    """
    return tuple(n==1 for n in shape)

def tensor(object, name=None, dtype=None):
    broadcastable = None
    if isinstance(object, tuple):
        shape = object
    elif hasattr(object, 'shape'):
        shape = object.shape
        if dtype is None: dtype = object.dtype
    elif hasattr(object, 'broadcastable'):
        if dtype is None: dtype = object.dtype
        broadcastable = object.broadcastable
    if not cf.use_theano:
        # If `shape` is undefined at this point, we gave a wrong argument
        return np.array(shape, dtype=dtype)
    else:
        if broadcastable is None: broadcastable = shape_to_broadcast(shape)
        return getT().tensor(dtype, broadcastable, name=name)

######################
# Shared variable constructor

class ShimmedShared(np.ndarray):
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
        # Allow mixing of ShimmedShared and Theano variables
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
                # a reference to this ShimmedShared variable exists elsewhere,
                # and we try to access it after the resize. This is the kind
                # of thing you shouldn't do anyway with Theano variables.
            self[:] = new_value
        except IndexError:
            # Scalars will fail on the above
            assert(isscalar(new_value))
                # np.isscalar will fail on 0-dim arrays; isscalar works
            self = super(ShimmedShared, self).__setitem__(None, new_value)

    def eval(self, inputs_to_values=None):
        return self.get_value()

    @property
    def broadcastable(self):
        """For Numpy arrays, an axis is broadcastable iff it has length one."""
        return tuple(s==1 for s in self.shape)

cf.add_terminating_types([ShimmedShared])

def shared(value, name=None, strict=False, allow_downcast=None, symbolic=True,
           **kwargs):
    """
    In contrast to Theano's `shared()`, the broadcast pattern is set to be compatible
    with NumPy's behaviour; i.e., any axis in `value` with dimension 1 is considered
    broadcastable.
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
        return ShimmedShared(value, name, strict, allow_downcast, **kwargs)

######################
# Interchangeable set_subtensor
def set_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if 'theano' in sys.modules and (isinstance(x, _gettheano().gof.Variable)
                                    or isinstance(y, _gettheano().gof.Variable)):
        return _getT().set_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert(x.base is not None)
            # Ensure that x is a view of another ndarray
        assert(x.shape == y.shape)
        x[:] = y
        return x.base

def inc_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if 'theano' in sys.modules and (isinstance(x, _gettheano().gof.Variable)
                                    or isinstance(y, _gettheano().gof.Variable)):
        return T.inc_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert(x.base is not None)
            # Ensure that x is a view of another ndarray
        assert(x.shape == y.shape)
        x[:] += y
        return x.base

# TODO: Deprecate: numpy arrays have ndim
def get_ndims(x):
    if cf.use_theano and isinstance(x, theano.gof.Variable):
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
    if is_theano_variable(x):
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
    if is_theano_variable(x):
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

load(load_theano=False)
    # By default, don't load Theano

#######################
# NumPy functions

def all(x):
    if is_theano_object(x):
        return T.all(x)
    else:
        return np.all(x)
def arange(start, stop=None, step=1, dtype=None):
    if is_theano_object(start, stop, step, dtype):
        return T.arange(start, stop, step, dtype)
    else:
        return np.arange(start, stop, step, dtype)
def broadcast_to(array, shape, subok=False):
    if is_theano_object(array, shape):
        return T.ones(shape) * array
    else:
        return np.broadcast_to(array, shape, subok)
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
def dot(x, y):
    if is_theano_object(x) or is_theano_object(y):
        return T.dot(x, y)
    else:
        return np.dot(x, y)
def exp(x):
    if is_theano_object(x):
        return T.exp(x)
    else:
        # if isinstance(x, ShimmedShared):
        #     x = x.get_value()
        return np.exp(x)
def gammaln(x):
    if is_theano_object(x):
        return T.gammaln(x)
    else:
        return sp.special.gammaln(x)
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
def min(x):
    if is_theano_object(x):
        return T.min(x)
    else:
        return np.min(x)
def max(x):
    if is_theano_object(x):
        return T.max(x)
    else:
        return np.max(x)
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
            if not isinstance(axis, collections.Iterable):
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
