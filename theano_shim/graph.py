"""
Shimmed Graph utilities:
   graph compilation, traversal, listing inputs...
"""
import collections
import itertools
from . import core
from .config import config as cf

class TooCostly(Exception):
    pass

######################
# Graph manipulation

def clone(output, replace=None, *args, **kwargs):
    """
    Use as theano.clone().
    TODO: Something useful with non-symbolic output ?
    """
    if not core.is_theano_variable(output):
        raise ValueError("`shim.graph.clone()` is undefined for non-symbolic outputs")
    return core.theano.clone(output, replace, *args, **kwargs)

#####################
# Graph compilation

def compile(inputs, outputs, *args, **kwargs):
    """
    Use as theano.function().
    TODO: Something useful with non-symbolic output ?
    """
    if not any(core.is_theano_variable(arg)
               for arg in itertools.chain([inputs, outputs], args, kwargs.values())):
        raise ValueError("`shim.graph.function()` is undefined for non-symbolic outputs")
    return core.theano.function(inputs, outputs, *args, **kwargs)

def eval(expr, inputs=None, max_cost=10):
    """
    Obtain a numerical value by evaluating an expression's graph.

    If expr is shared (or shimmed shared) variable, its `get_value()` method is
    called and the result returned.
    Otherwise, the result of `expr.eval(inputs)` is  if Theano is not loaded, `expr` is simply returned

    Parameters
    ----------
    expr: Theano graph | slice | iterable (TODO)
        The theano expression we want to evaluate. If a slice, each component
        of the slice ('start', 'stop', 'step') is evaluated.
        TODO: iterable
    inputs: dict (optional; default: {})
        Dictionary of inputs to be past to `eval`.
    max_cost: int (optional; default: 10)
        Maximum allowable 'cost' of the expression: if `expr` is estimated to
        be more costly than this, an error is raised instead. The value `None`
        deactivates this check.
        Cost is estimated by a proxy, specifically the number of ancestors to
        `expr` in the Theano graph. By default, only expressions with 10 ancestors
        or less are evaluated.

    Returns
    -------
    Pure Python variable of same type as `expr`.
    """
    # Bypassing code paths
    if core.isshared(expr):
        return expr.get_value()
    elif not core.is_theano_object(expr):
        return expr
    elif isinstance(expr, slice):
        return slice(eval(expr.start), eval(expr.stop), eval(expr.step))

    # "Standard" code path
    cost = len(core.gettheano().gof.graph.ancestors([expr]))
    if max_cost is not None and cost > max_cost:
        raise TooCostly("Expression has {} ancestors, which exceeds the limit of {}."
                        .format(cost, max_cost))
    if inputs is None:
        inputs = {}
    return expr.eval(inputs)

######################
# Graph inspection

def is_computable(varlist, with_inputs=None):
    """
    Returns True if the variables in varlist can be numerically evaluated
    using only the inputs in `with_inputs`. In other words, the computational
    graph associated to varlist is composed only of constants and shared variables,
    along with the symbolic variables in `with_inputs`.
    If varlist is not a Theano graph, it is always computable.
    """
    if ( not isinstance(varlist, collections.Iterable)
         or isinstance(varlist, str) ):
        raise ValueError("theano_shim.is_computable requires a list as first argument.")
    if with_inputs is None:
        with_inputs = []
    computable = True
    for var in varlist:
        if isinstance(var, slice):
            # Must come before `is_theano_variable`
            if not is_computable([var.start, var.stop, var.step]):
                computable = False
                break
        elif core.is_theano_variable(var): # Required because varlist may contain non-Theano objects
            if core.is_theano_variable( set(core.gettheano().gof.graph.inputs([var])).difference(with_inputs) ):
                computable = False
                break
    return computable

def inputs(varlist, *args, **kwargs):
    """
    Wrapper for theano.gof.graph.inputs.
    Returns an empty list for non symbolic variables
    """
    if ( not isinstance(varlist, collections.Iterable)
         or isinstance(varlist, str) ):
        raise ValueError("theano_shim.is_computable requires a list as first argument.")
    if core.is_theano_object(varlist):
        return core._gettheano().gof.graph.inputs(varlist, *args, **kwargs)
    else:
        return []

def symbolic_inputs(varlist, *args, **kwargs):
    return [v for v in inputs(varlist, *args, **kwargs) if core.is_theano_variable(v)]
