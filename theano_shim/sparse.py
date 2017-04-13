import collections
import scipy as sp
import scipy.sparse

from . import core
from .core import is_theano_object

######################
# Sparse matrices
# Theano just wraps the Scipy sparse matrices, so there's not much to do here
# However, Theano only supports csr and csc matrices. For others we would need
# to write our own wrappers.

def csr_matrix(name, shape, dtype=None):
    if dtype is None:
        if core.use_theano:
            dtype = theano.config.floatX
        else:
            dtype = 'float32'
    if core.use_theano:
        return theano.sparse.csr_matrix(name=name, dtype=dtype)
    else:
        return sp.sparse.csr_matrix(shape, dtype=dtype)

def csc_matrix(name, shape, dtype=None):
    if dtype is None:
        if core.use_theano:
            dtype = theano.config.floatX
        else:
            dtype = 'float32'
    if core.use_theano:
        return theano.sparse.csc_matrix(name=name, dtype=dtype)
    else:
        return sp.sparse.csc_matrix(shape, dtype=dtype)

def csr_from_dense(x):
    if is_theano_object(x):
        return theano.sparse.csr_from_dense(x)
    else:
        return sp.sparse.csr_matrix(x)
def csc_from_dense(x):
    if is_theano_object(x):
        return theano.sparse.csc_from_dense(x)
    else:
        return sp.sparse.csc_matrix(x)

def coo_matrix(name, shape, dtype=None):
    if dtype is None:
        if core.use_theano:
            dtype = theano.config.floatX
        else:
            dtype = 'float32'

    if isinstance(name, tuple):
        # 'name' is actually the data
        data, (row, col) = name
        if is_theano_object(data, row, col):
            raise NotImplementedError
        else:
            return sp.sparse.coo_matrix((data, (row, col)), shape)

    else:
        if core.use_theano:
            raise NotImplementedError("Theano doesn't support coo_matrix natively, and a wrapper has not yet been implemented in shim.")
        else:
            return sp.sparse.coo_matrix(shape, dtype=dtype)

def hstack(blocks, format=None, dtype=None):
    if ( ( isinstance(blocks, collections.abc.Iterable)
           and is_theano_object(*blocks) )
         or is_theano_object(blocks) ):
        return T.sparse.hstack(blocks, format, dtype)
    else:
        return sp.sparse.hstack(blocks, format, dtype)

def vstack(blocks, format=None, dtype=None):
    if ( ( isinstance(blocks, collections.abc.Iterable)
           and is_theano_object(*blocks) )
         or is_theano_object(blocks) ):
        return T.sparse.vstack(blocks, format, dtype)
    else:
        return sp.sparse.vstack(blocks, format, dtype)
