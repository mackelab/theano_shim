"""
Shimmed Graph utilities:
   graph compilation, traversal, listing inputs...
"""
import collections
import itertools
from . import core
from .config import config as cf

# Custom exceptions
class TooCostly(Exception):
    pass
# Theano exceptions
class _MissingInputError(Exception):
    pass
MissingInputError = _MissingInputError
def load_exceptions():
    if cf.use_theano:
        import theano.gof as gof
        MissingInputError = gof.MissingInputError
    else:
        MissingInputError = _MissingInputError

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

def eval(expr, givens=None, max_cost=10, if_too_costly='raise', inputs=None):
    """
    Obtain a numerical value by evaluating an expression's graph.

    If expr is shared (or shimmed shared) variable, its `get_value()` method is
    called and the result returned.
    Otherwise, a function is compiled and evaluated. `givens` can be used
    to replace symbolic and shared variables during compilation.

    If it's not a symbolic expression at all, `expr` is simply returned.

    Parameters
    ----------
    expr: Theano graph | slice | iterable (TODO)
        The theano expression we want to evaluate. If a slice, each component
        of the slice ('start', 'stop', 'step') is evaluated.
        TODO: iterable
    givens: dict (optional; default: {})
        Dictionary of inputs to be past to `eval`.
    max_cost: int (optional; default: 10)
        Maximum allowable 'cost' of the expression: if `expr` is estimated to
        be more costly than this, an error is raised instead. The value `None`
        deactivates this check.
        Cost is estimated by a proxy, specifically the number of ancestors to
        `expr` in the Theano graph. By default, only expressions with 10 ancestors
        or less are evaluated.
    if_too_costly: 'ignore'  |  'raise'
        What to do if an expression is too costly to compute.
        'ignore': do nothing, return symbolic expression.
        'raise' : raise a TooCostly exception.
    inputs: Deprecated
        Synonym for `givens`.

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
        if if_too_costly == 'raise':
            raise TooCostly("Expression has {} ancestors, which exceeds the "
                            "limit of {}.".format(cost, max_cost))
        else:
            return expr
    else:
        if inputs is None: inputs = {}  # TODO: Remove `inputs`
        if givens is None: givens = {}
        givens = {**inputs, **givens}
        for k, v in givens.items():
            givens[k] = core.cast(v, k.dtype)
        f = core.gettheano().function(
            [], expr, givens=givens, on_unused_input='ignore')
        return f()

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
        raise ValueError("theano_shim.graph.inputs requires a list as first argument.")
    if core.is_theano_object(varlist):
        return core._gettheano().gof.graph.inputs(varlist, *args, **kwargs)
    else:
        return []

def variables(i, o=None):
    """
    Wrapper for theano.gof.graph.variables.
    Returns an empty list for non symbolic variables
    If given only one argument, returns all variables it depends on.
    """
    if o is None:
        o = i
        i = inputs(o)
    if ( not isinstance(i, collections.Iterable)
         or isinstance(i, str) ):
        raise ValueError(
            "Arguments to theano_shim.graph.variables must be lists.")
    if core.is_theano_object(i, o):
        return core._gettheano().gof.graph.variables(i, o)
    else:
        return []

def symbolic_inputs(varlist, *args, **kwargs):
    return [v for v in inputs(varlist, *args, **kwargs) if core.is_theano_variable(v)]

def is_same_graph(var1, var2, givens=None, debug=False):
    """
    Wraps theano.gof.graph.is_same_graph().
    Returns True if `var1` and `var2` perform the same computation.
    If either `var1` or `var2` is not a graph object, returns the result of
    `var1 == var2`.
    """
    if not (core.is_theano_object(var1) and core.is_theano_object(var2)):
        return var1 == var2
    else:
        return core._gettheano().gof.graph.is_same_graph(var1, var2,
                                                         givens, debug)
