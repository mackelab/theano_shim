from .core import *
from . import core
from . import graph
from . import symbolic

def load_theano():
    load(True)

def load(load_theano=True, reraise=False):
    core.load(load_theano, reraise)
    graph.load_exceptions()
