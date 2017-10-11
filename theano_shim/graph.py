"""
Shimmed Graph utilities:
   graph compilation, traversal, listing inputs...
"""
from . import core
from . import config as cf

class TooCostly(Exception):
    pass

def eval(expr, inputs=None, max_cost=10):
    """
    Obtain a numerical value by evaluating an expression's graph.

    If expr is shared (or shimmed shared) variable, its `get_value()` method is
    called and the result returned.
    Otherwise, the result of `expr.eval(inputs)` is  if Theano is not loaded, `expr` is simply returned

    Parameters
    ----------
    expr: Theano graph
        The theano expression we want to evaluate.
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

    # "Standard" code path
    cost = len(core.theano.gof.graph.ancestors([expr]))
    if max_cost is not None and cost > max_cost:
        raise TooCostly("Expression has {} ancestors, which exceeds the limit of {}."
                        .format(cost, max_cost))
    if inputs is None:
        inputs = {}
    return expr.eval(inputs)

# def inputs(x, *args, **kwargs):
#     """
#     Wrapper for theano.gof.graph.inputs.
#     Returns an empty list if Theano is not loaded.
#     """
#     if not cf.use_theano:
#         return []
#     else:
#         return core.theano.gof.graph.inputs(x, *args, **kwargs)
