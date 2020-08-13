from .core import *
from . import core
from . import graph
from . import symbolic

def load_theano():
    load('theano')

def load(library='theano', reraise=False):
    core.load(library, reraise)
    graph.load_exceptions()

# Common functions which are worth having in global namespace
eval = graph.eval
