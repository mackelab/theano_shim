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

import os
import logging
import builtins
import collections
import numpy as np
import scipy as sp
import scipy.signal

logger = logging.getLogger('theano_shim')
logger.setLevel(logging.INFO)
_fh = logging.FileHandler("theano_shim_" + str(os.getpid()) + ".log", mode='w')
_fh.setLevel(logging.DEBUG)
_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
_logging_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_fh.setFormatter(_logging_formatter)
_ch.setFormatter(_logging_formatter)
logger.addHandler(_fh)
logger.addHandler(_ch)

use_theano = False
inf = np.inf

theano_updates = {}
    # Stores a Theano update dictionary. See below for use

lib = None
    # DEPRECATION WARNING: lib will soon be removed
RandomStreams = None
#######################
# Initialization function.
# Import the appropriate numerical library into this namespace,
# so we can make calls like `lib.exp`

def load_theano():
    load(True)

def load(load_theano = False, reraise=False):
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
    global use_theano
    global theano, T, inf, lib, RandomStreams

    if load_theano:
        try:
            import theano
        except ImportError:
            logger.error("The theano library was not found.")
            use_theano = False
            if reraise:
                raise
        else:
            use_theano = True

    if use_theano:
        import theano.tensor as T
        import theano.tensor as lib
        import theano.ifelse
        import theano.tensor.shared_randomstreams  # CPU only
        #from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU & GPU

        inf = 1e12
        RandomStreams = theano.tensor.shared_randomstreams.RandomStreams

    else:
        import numpy as lib
        inf = np.inf
        RandomStreams = ShimmedRandomStreams

##########################
# Managing theano updates

def add_update(variable, value):
    logger.info("Adding Theano update : {} -> {}".format(variable.name, str(value)))
    if variable in self.theano.updates:
        raise ValueError("Cannot update the same shared variable twice. "
                         "It should be used in a `theano.function` call, and then "
                         "cleared with `shim.theano_reset()` before being "
                         "updated again.")
    if not is_shared_variable(variable):
        raise ValueError("The updates mechanism only applies to shared variables.")

    self.theano_updates[variable] = value

def add_updates(updates):
    """
    Parameters
    ----------
    updates: dict or iterable
        Either a dictionary of `variable:value` pairs, or an iterable of
        `(variable, value)` tuples.
    """
    if isinstance(updates, dict):
        for key, val in updates.items():
            add_update(key, val)
    else:
        for key, val in updates:
            add_update(key, val)

def theano_reset():
    logger.info("Clearing Theano updates")
    theano_updates = {}

#######################
# Assert equivalent
def check(stmt):
    """Check is a library-aware wrapper for assert.
    If stmt is a Theano variable, the behaviour depends on whether
    theano.config.compute_test_value:
        - If it is 'off', `check` is a no-op
        - Otherwise, use the test values to evaluate the assert
    """
    if not use_theano or not isinstance(stmt, theano.gof.Variable):
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
    if use_theano and isinstance(var, T.sharedvar.SharedVariable):
        retval = var.get_value()
    elif use_theano and isinstance(var, theano.gof.Variable):
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
    type_str: string or iterable
        If `obj` is of this type, the function returns True,
        otherwise it returns False. Valid values of `type_str`
        are those expected for a dtype. Examples are:
        - 'int', 'int32', etc.
        - 'float', 'float32', etc.
        `type_str` can also be an iterable of aforementioned
        strings. Function will return True if `obj` is of any
        of the specifed types

    Returns
    -------
    bool
    """
    # Wrap type_str if it was not passed as an iterable
    if isinstance(type_str, str):
        type_str = [type_str]
    # Check type
    if not use_theano or not isinstance(obj, theano.gof.Variable):
        return any(ts in str(np.asarray(obj).dtype) for ts in type_str)
            # We cast to string to be consistent with Theano, which uses
            # strings for it's dtypes
    else:
        return any(ts in obj.dtype for ts in type_str)

def is_theano_variable(var):
    return use_theano and isinstance(var, theano.tensor.TensorVariable)
def is_theano_object(obj):
    return use_theano and isinstance(obj, theano.gof.Variable)
def is_shared_variable(var):
    if use_theano:
        return isinstance(var, T.sharedvar.SharedVariable)
    else:
        return isinstance(var, ShimmedShared)

#######################
# Functions to cast to an integer variable
def cast_int8(x):
    if is_theano_object(x):
        return T.cast(x, 'int8')
    else:
        return np.int8(x)
def cast_int16(x):
    if use_theano and isinstance(x, theano.gof.Variable):
        return T.cast(x, 'int16')
    else:
        return np.int16(x)
def cast_int32(x):
    if use_theano and isinstance(x, theano.gof.Variable):
        return T.cast(x, 'int32')
    else:
        return np.int32(x)
def cast_int64(x):
    if use_theano and isinstance(x, theano.gof.Variable):
        return T.cast(x, 'int64')
    else:
        return np.int64(x)

#####################
# Simple convenience functions
def round(x):
    try:
        res = x.round()  # Theano variables have a round method
    except AttributeError:
        res = round(x)
    return res

def asvariable(x, dtype=None):
    if use_theano:
        # No `isinstance` here: the point is to cast to a Theano variable
        if dtype is not None:
            return T.cast(T.as_tensor_variable(x), dtype)
        else:
            return T.as_tensor_variable(x)
    else:
        return np.asarray(x, dtype=dtype)

def asarray(x, dtype=None):
    if use_theano and isinstance(x, theano.gof.Variable):
        if dtype is not None:
            return T.cast(T.as_tensor_variable(x), dtype)
        else:
            return T.as_tensor_variable(x)
    else:
        return np.asarray(x, dtype=dtype)

def isscalar(x):
    arrayed_x = asarray(x)
    return asarray(x).ndim == 0 and arrayed_x.dtype != 'object'

def isarray(x):
    return hasattr(x, 'ndim')

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


def flatten(x, outdim=1):
    if use_theano and isinstance(x, theano.gof.Variable):
        return T.flatten(x, outdim)
    else:
        outshape = x.shape[:outdim-1] + (np.prod(x.shape[outdim-1:]), )
        return x.reshape(outshape)

#####################
# Convenience function for max / min

def largest(*args):
    """Element-wise max operation."""
    assert(len(args) >= 2)
    if use_theano and any(isinstance(arg, theano.gof.Variable) for arg in args):
        return T.largest(*args)
    else:
        retval = np.maximum(args[0], args[1])
        for arg in args[2:]:
            retval = np.maximum(retval, arg)
        return retval

def smallest(*args):
    """Element-wise min operation."""
    assert(len(args) >= 2)
    if use_theano and any(isinstance(arg, theano.gof.Variable) for arg in args):
        return T.smallest(*args)
    else:
        retval = np.minimum(args[0], args[1])
        for arg in args[2:]:
            retval = np.minimum(retval, arg)
        return retval

def abs(x):
    if use_theano and isinstance(x, theano.gof.Variable):
        if x.ndim == 2:
            return __builtins__['abs'](x)
        else:
            # Theano requires 2D objects for abs
            shape = x.shape
            return __builtins__['abs'](add_axes(x.flatten())).reshape(shape)
    else:
        return __builtins__['abs'](x)
#####################
# Set random functions

class ShimmedRandomStreams:
    def __init__(self, seed=None):
        np.random.seed(seed)

    def normal(self, size=(), avg=0.0, std=1.0, ndim=None):
        return np.random.normal(loc=avg, scale=std, size=size)

    def uniform(self, size=(), low=0.0, high=1.0, ndim=None):
        return np.random.uniform(low, high, size)

    def binomial(self, size=(), n=1, p=0.5, ndim=None):
        return np.random.binomial(n, p, size)

################################################
# Define Theano placeins, which execute
# equivalent Python code if Theano is not used.
# Many Python versions take useless arguments,
# to match the signature of the Theano version.
################################################

######################
# Logical and comparison operators

def lt(a, b):
    if (use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.lt(a, b)
    else:
        return a < b
def le(a, b):
    if (use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.le(a, b)
    else:
        return a <= b
def gt(a, b):
    if (use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.gt(a, b)
    else:
        return a > b
def ge(a, b):
    if (use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.ge(a, b)
    else:
        return a >= b
def eq(a, b):
    if (use_theano and (isinstance(a, theano.gof.Variable)
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
    if use_theano and isinstance(a, builtins.bool):
        return np.int8(a)
    elif use_theano:
        return a
    else:
        return builtins.bool(a)

def and_(a, b):
    # Special case scalars so they don't return length 1 arrays
    if isscalar(a) and isscalar(b):
        return bool(bool(a) * bool(b))

    # matrix function
    if (use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.and_(a, b)
    else:
        return np.logical_and(a, b)

def or_(a, b):
    # Special case scalars so they don't return length 1 arrays
    if isscalar(a) and isscalar(b):
        return bool(bool(a) + bool(b))

    # matrix function
    if (use_theano and (isinstance(a, theano.gof.Variable)
                        or isinstance(b, theano.gof.Variable))):
        return T.or_(a, b)
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
    if (use_theano and (isinstance(condition, theano.gof.Variable)
                        or isinstance(then_branch, theano.gof.Variable)
                        or isinstance(else_branch, theano.gof.Variable))):
        # Theano function
        if outshape is None:
            return theano.ifelse.ifelse(condition, then_branch,
                                        else_branch, name)
        else:
            return theano.ifelse.ifelse(condition, then_branch.reshape(outshape),
                                        else_branch.reshape(outshape), name)
    else:
        # Python function
        if condition:
            return then_branch
        else:
            return else_branch

def switch(cond, ift, iff):
    if (use_theano and (isinstance(cond, theano.gof.Variable)
                        or isinstance(ift, theano.gof.Variable)
                        or isinstance(iff, theano.gof.Variable))):
        return T.switch(cond, ift, iff)
    else:
        return np.where(cond, ift, iff)


######################
# Shared variable constructor

class ShimmedShared(np.ndarray):
    # See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    # for indications on subclassing ndarray

    def __new__(cls, value, name=None, strict=False, allow_downcast=None, **kwargs):
        obj = np.asarray(value).view(cls).copy()
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)

    # We are emulating theano.shared, where different instances
    # are considred distinct
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)

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
            assert(np.isscalar(new_value))
            self = super(ShimmedShared, self).__setitem__(None, new_value)

def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    if use_theano:
        return theano.shared(value, name, strict, allow_downcast, **kwargs)
    else:
        return ShimmedShared(np.asarray(value), name, strict, allow_downcast, **kwargs)


######################
# Interchangeable set_subtensor
def set_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if use_theano and (isinstance(x, theano.gof.Variable)
                       or isinstance(y, theano.gof.Variable)):
        return T.set_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert(x.base is not None)
            # Ensure that x is a view of another ndarray
        x[:] = y
        return x.base

def inc_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if use_theano and (isinstance(x, theano.gof.Variable)
                       or isinstance(y, theano.gof.Variable)):
        return T.inc_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert(x.base is not None)
            # Ensure that x is a view of another ndarray
        x[:] += y
        return x.base

# TODO: Deprecate: numpy arrays have ndim
def get_ndims(x):
    if use_theano and isinstance(x, theano.gof.Variable):
        return x.ndim
    else:
        return len(x.shape)

######################
# Axis manipulation functions
# E.g. to treat a scalar as a 1x1 matrix

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
        - 'before', 'left' turns a 1D vector into a row vector. (Default)
        - 'after', 'right' turns a 1D vector into a column vector.
        - 'before last' adds axes to the second-last position.
          Equivalent to 'left' on 1D vectors.'.
        - An integer adds the axes before this position
            + 0 : equivalent to 'before'
            + -1 : equivalent to 'before last'
            + `x.ndim` : equivalent to 'after'
    """
    if use_theano and isinstance(x, theano.gof.Variable):
        if pos in ['left', 'before']:
            shuffle_pattern = ['x']*num
            shuffle_pattern.extend(range(x.ndim))
        elif pos  in ['right', 'after']:
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
        if pos in ['left', 'before']:
            return x.reshape( (1,)*num + x.shape )
        elif pos in ['right', 'after']:
            return x.reshape( x.shape + (1,)*num )
        elif pos == 'before last':
            return x.reshape( x.shape[:-1] + (1,)*num + x.shape[-1:] )
        else:
            try:
                return x.reshape( x.shape[:pos] + (1,)*num + x.shape[pos:] )
            except ValueError:
                raise ValueError("Unrecognized argument {} for pos.".format(pos))

def moveaxis(a, source, destination):
    if use_theano and isinstance(x, theano.gof.Variable):
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
    elif is_shared_variable(array):
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
            for i, (w, v) in enumerate(zip(pad_width, vals)):
                if (w[0] != 0 or w[1] != 1):
                    shape1 = array_shape[:i] + (w[0],) + array_shape[i+1:]
                    shape2 = array_shape[:i] + (w[1],) + array_shape[i+1:]
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
    check((n >= 0).all())
    if is_theano_object(n):
        return T.gamma(n+1)
    else:
        return sp.misc.factorial(n, exact)


########################
# Wrapper for discrete 1D convolutions

# TODO: Use fftconvolve if ~500 time bins or more

def conv1d(history_arr, discrete_kernel_arr, mode='valid'):
    """
    Applies the convolution to each component of the history
    and stacks the result into an array

    Parameters
    ----------
    history: ndarray | theano.tensor
        Return value from indexing history[begin1:end1],
        where history is a Series instance with shape (M,)
    discrete_kernel: ndarray | theano.tensor
        Return value from indexing discrete_kernel[begin2:end2],
        where discret_kernel is a Series instance with shape (M, M)
        obtained by calling history.discretize_kernel.

    Returns
    -------
    ndarray:
        Result has shape (M, M)
    """

    check(len(history_arr.shape) == 2)
    output_shape = discrete_kernel_arr.shape[1:]
    if (discrete_kernel_arr.ndim == 2):
        # Algorithm assumes a "to" axis on the kernel. Add it.
        add_axes(discrete_kernel_arr, 1, 'before last')
    else:
        check(discrete_kernel_arr.ndim == 3)

    # Convolutions leave the time component on the inside, but we want it on the outside
    # So we do the iterations in reverse order, and flip the result with transpose()
    # The result is indexed as [tidx][to idx][from idx]
    if use_theano:
        # We use slices from_idx:from_idx+1 because conv2d expects 2D objects
        # We then index [:,0] to remove the spurious dimension
        result = T.stack(
                  [ T.stack(
                       [ T.signal.conv.conv2d(history_arr[:, from_idx:from_idx+1 ],
                                              discrete_kernel_arr[:, to_idx, from_idx:from_idx+1 ],
                                              image_shape = (len(history_arr._tarr), 1),
                                              filter_shape = (len(kernel_arr._tarr), 1),
                                              border_mode = mode)[:,0]
                         for to_idx in T.arange(discrete_kernel_arr.shape[1]) ] )
                       for from_idx in T.arange(discrete_kernel_arr.shape[2]) ] ).T
    else:
        result = np.stack(
                  [ np.stack(
                       [ scipy.signal.convolve(history_arr[:, from_idx ],
                                            discrete_kernel_arr[:, to_idx, from_idx ],
                                            mode=mode)
                         for to_idx in np.arange(discrete_kernel_arr.shape[1]) ] )
                       for from_idx in np.arange(discrete_kernel_arr.shape[2]) ] ).T

    return result.reshape(result.shape[0:1] + output_shape)



################################
# Module initialization

load(load_theano=False)
    # By default, don't load Theano

#######################
# Straight redirects to NumPy/Theano

def all(x):
    if is_theano_object(x):
        return T.all(x)
    else:
        return np.all(x)
def concatenate(tensor_list, axis=0):
    if any(is_theano_object(x) for x in tensor_list):
        return T.concatenate(tensor_list, axis=0)
    else:
        return np.concatenate(tensor_list, axis=0)
def cos(x):
    if is_theano_object(x):
        return T.cos(x)
    else:
        return np.cos(x)
def exp(x):
    if is_theano_object(x):
        return T.exp(x)
    else:
        return np.exp(x)
def log(x):
    if is_theano_object(x):
        return T.log(x)
    else:
        return np.log(x)
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
def sum(x, axis=None, dtype=None, acc_dtype=None, keepdims=np._NoValue):
    if is_theano_object(x):
        result = T.sum(x, axis, dtype, acc_dtype)
        if keepdims:
            if not isinstance(axis, collections.Iterable):
                axes = [axis]
            else:
                axes = sorted(axis)
            for axis in axes:
                result = add_axes(result, pos=axis)
        return result
    else:
        return np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
def tile(x, reps, ndim=None):
    if is_theano_object(x):
        return T.tile(x, reps, ndim)
    else:
        return np.tile(x, reps)

# The following is code that could be used for automatic
# redirects on a class
# #######################
# # Default behaviour is to redirect to NumPy or Theano
# # if a particular attribute is not already defined

# class _LibAttribute:
#     def __init__(self, name):
#         try:
#             self.npattr = getattr(np, name)
#             self.libattr = getattr(shim.lib, name)
#         except AttributeError:
#             raise AttributeError("theano_shim does not define '{}'.".format(name))

#     def __call__(self, *args):
#         try:
#             return self.npattr(*args)
#         except TypeError:
#             return self.libattr(*args)

# def __getattr__(self, name):
#     return _LibAttribute(name)
