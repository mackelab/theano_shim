import collections.abc
import types
import scipy as sp
import scipy.sparse

from . import core
from .core import is_graph_object
from .config import config as cf
from .config import _gettheano, _getT

# Add to TerminatingTypes
cf.add_terminating_types([sp.sparse.spmatrix])

# TODO: move to core
from functools import wraps
def forcable_symbolic(f):
    @wraps(f)
    def wrapper(*args, symbolic=None, **kwargs):
        _symbolic = is_graph_object(*args, *kwargs.values())
        if symbolic is None:
            symbolic = _symbolic
        elif symbolic is False and _symbolic is True:
            raise ValueError("Cannot force a symbolic variable to be numeric.")
        return f(*args, **kwargs, symbolic=symbolic)
    return wrapper

######################
# Utilities

# E.g. kernel, analyze need to check if an array is sparse, and we don't want
# to load the module just to allow that check, so the checking functions are
# defined in `core`
issparse = core.issparse
isspsparse = core.isspsparse

######################
# Sparse matrices
# In theory, Theano just wraps the Scipy sparse matrices, so there shouldn't
# be much to do here.
# However, the Theano sparse matrices are in fact arrays (=> element-wise
# multiplication instead of matrix multiplication), so for consistency
# we define a sparse matrix wrapper which treats * as element-wise.

# Also, Theano only supports CSR and CSC arrays.

def _remap_mul(f):
    # The sparse methods like .sum() expect __mul__ to be matrix
    # multiplication, so we temporarily remap __mul__ for all method calls
    @wraps(f)
    def wrapper(self, *a, **kw):
        remap = self.__mul__ == self._array_mul  # `is` doesn't work w/ methods
        if remap:  # Only remap if it hasn't already been done
            self.__mul__ = super(type(self), self).__mul__
            self.__rmul__ = super(type(self), self).__rmul__
        v = f(self, *a, **kw)
        if remap:
            self.__mul__ = self._array_mul
            self.__rmul__ = self._array_rmul
        return v
    return wrapper

# Names need to start with csr, csc  (https://stackoverflow.com/a/24509511)
# TODO: Mixin class, to make csc & coo wrappers
matT = sp.sparse.csr_matrix
class csr_matrix_wrapper(matT):
    """
    Wrapper which replaces the * operator of a sparse matrix (which acts
    as a dot product) with the more standard element-wise multiplication.
    This makes it consistent with Theano's * operator.
    """
    def _array_mul(self, other):
        return self.multiply(other)
    def _array_rmul(self, other):
        return self.multiply(other)
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.__mul__ = self._array_mul
        self.__rmul__ = self._array_rmul
    # We redefine anything which may call __mul__ internally, since in that
    # case it would expect matrix multiplication
    arcsin         = _remap_mul(matT.arcsin)
    acrsinh        = _remap_mul(matT.arcsinh)
    arctan         = _remap_mul(matT.arctan)
    arctanh        = _remap_mul(matT.arctanh)
    count_nonzero  = _remap_mul(matT.count_nonzero)
    deg2rad        = _remap_mul(matT.deg2rad)
    dot            = _remap_mul(matT.dot)
    expm1          = _remap_mul(matT.expm1)
    log1p          = _remap_mul(matT.log1p)
    mean           = _remap_mul(matT.mean)
    multiply       = _remap_mul(matT.multiply)
    power          = _remap_mul(matT.power)
    rad2deg        = _remap_mul(matT.rad2deg)
    sin            = _remap_mul(matT.sin)
    sinh           = _remap_mul(matT.sinh)
    sqrt           = _remap_mul(matT.sqrt)
    sum            = _remap_mul(matT.sum)
    sum_duplicates = _remap_mul(matT.sum_duplicates)
    tan            = _remap_mul(matT.tan)
    tanh           = _remap_mul(matT.tanh)

# class csc_matrix_wrapper(sp.sparse.csc_matrix):
#     """
#     Wrapper which replaces the * operator of a sparse matrix (which acts
#     as a dot product) with the more standard element-wise multiplication.
#     This makes it consistent with Theano's * operator.
#     """
#     def _array_mul(self, other):
#         return self.multiply(other)
#     def _array_rmul(self, other):
#         return self.multiply(other)
#     def __init__(self, *a, **kw):
#         self.__mul__ = object.__getattribute__(self, '_array_mul')
#         self.__rmul__ = object.__getattribute__(self, '_array_rmul')
#     def __getattribute__(self, *a, **kw):
#         # The sparse methods like .sum() expect __mul__ to be matrix
#         # multiplication, so we temporarily remap __mul__ for all method calls
#         self.__mul__ = super().__mul__
#         self.__rmul__ = super().__rmul__
#         super().__getattribute__(*a, **kw)
#         self.__mul__ = object.__getattribute__(self, '_array_mul')
#         self.__rmul__ = object.__getattribute__(self, '_array_rmul')

# class coo_matrix_wrapper(sp.sparse.coo_matrix):
#     """
#     Wrapper which replaces the * operator of a sparse matrix (which acts
#     as a dot product) with the more standard element-wise multiplication.
#     This makes it consistent with Theano's * operator.
#     """
#     def __mul__(self, other):
#         return self.multiply(other)
#     def __rmul__(self, other):
#         return self.multiply(other)

def csr_matrix(name, shape, dtype=None, symbolic=None):
    if dtype is None:
        dtype = cf.floatX
    if symbolic is None:
        symbolic = cf.use_theano
    if symbolic:
        return _gettheano().sparse.csr_matrix(name=name, dtype=dtype)
    else:
        return csr_matrix_wrapper(shape, dtype=dtype)

def csc_matrix(name, shape, dtype=None, symbolic=None):
    if dtype is None:
        dtype = cf.floatX
    if symbolic is None:
        symbolic = cf.use_theano
    if symbolic:
        return _gettheano().sparse.csc_matrix(name=name, dtype=dtype)
    else:
        return csc_matrix_wrapper(shape, dtype=dtype)

# ------------------------------------
# Theano interface for sparse matrices

@forcable_symbolic
def csr_from_dense(x, symbolic=None):
    if symbolic:
        return _gettheano().sparse.csr_from_dense(x)
    else:
        return csr_matrix_wrapper(x)
@forcable_symbolic
def csc_from_dense(x, symbolic=False):
    if symbolic:
        return _gettheano().sparse.csc_from_dense(x)
    else:
        return csc_matrix_wrapper(x)
@forcable_symbolic
def dense_from_sparse(x, symbolic=False):
    """
    Returns an array.
    (Contrast the `todense()` methods of `scipy.sparse`, which return a matrix.)
    """
    assert issparse(x)
    if symbolic:
        return _gettheano().sparse.dense_from_sparse(x)
    else:
        return x.todense().A

@forcable_symbolic
def CSR(data, indices, indptr, shape, symbolic=False):
    if symbolic:
        return _gettheano().sparse.CSR(data, indices, indptr, shape)
    else:
        return csr_matrix_wrapper((data, indices, indptr), shape)
@forcable_symbolic
def CSC(data, indices, indptr, shape, symbolic=False):
    if symbolic:
        return _gettheano().sparse.CSC(data, indices, indptr, shape)
    else:
        return csc_matrix_wrapper((data, indices, indptr), shape)

def csm_properties(x):
    assert issparse(x)
    if is_graph_object(x):
        return _gettheano().sparse.csm_properties(x)
    else:
        return x.data, x.indices, x.indptr, x.shape

# ------------------------------------

def coo_matrix(name, shape, dtype=None):
    if dtype is None:
        dtype = cf.floatX

    if isinstance(name, tuple):
        # 'name' is actually the data
        data, (row, col) = name
        if is_graph_object(data, row, col):
            raise NotImplementedError
        else:
            return coo_matrix_wrapper((data, (row, col)), shape)

    else:
        if cf.use_theano:
            raise NotImplementedError("Theano doesn't support coo_matrix "
                "natively, and a wrapper has not yet been implemented in shim.")
        else:
            return coo_matrix_wrapper(shape, dtype=dtype)

def hstack(blocks, format=None, dtype=None):
    if isinstance(blocks, types.GeneratorType):
        raise ValueError(
            "shim.sparse.hstack doesn't support generator arguments at the "
            "moment. Please use a list instead.")
    if ( ( isinstance(blocks, collections.abc.Iterable)
           and is_graph_object(*blocks) )
         or is_graph_object(blocks) ):
        return T.sparse.hstack(blocks, format, dtype)
    else:
        return sp.sparse.hstack(blocks, format, dtype)

def vstack(blocks, format=None, dtype=None):
    if isinstance(blocks, types.GeneratorType):
        raise ValueError(
            "shim.sparse.vstack doesn't support generator arguments at the "
            "moment. Please use a list instead.")
    if ( ( isinstance(blocks, collections.abc.Iterable)
           and is_graph_object(*blocks) )
         or is_graph_object(blocks) ):
        return T.sparse.vstack(blocks, format, dtype)
    else:
        return sp.sparse.vstack(blocks, format, dtype)
