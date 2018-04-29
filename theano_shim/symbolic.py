"""
Shimmed symbolic type
"""
from . import core
from .config import config as cf

######################
# Symbolic variable constructors
# TODO: Shimmed symbolic variable ?

def scalar(name, dtype='float64'):
    if cf.use_theano:
        return core.theano.tensor.scalar(name, dtype=dtype)
    else:
        raise RuntimeError("Must load Theano in order to create symbolic variables.")

def lscalar(name):
    return scalar(name, 'int64')
