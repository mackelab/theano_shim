"""
Shimmed Graph utilities:
   graph compilation, traversal, listing inputs...
"""
import sys
import collections
from typing import Optional, Union, Iterable, Generator, Collection
import itertools
import builtins
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
    "Called by `theano_shim.graph.load()`"
    global MissingInputError
    if cf.use_theano:
        import theano.graph.fg as fg
        MissingInputError = fg.MissingInputError
    else:
        MissingInputError = _MissingInputError

def reload():
    """Reload definitions according to cf.library."""
    # Add here any object who's definition changes depending on library
    global GraphExpression, GraphExpressionMeta

    if cf.library == 'numpy':
        GraphExpression = ShimmedGraphExpression
        GraphExpressionMeta = type
    elif cf.library == 'theano':
        GraphExpression = get_TheanoGraphExpression()
        # GraphExpressionMeta = core.gettheano().graph.utils.MetaObject
        GraphExpressionMeta = core.gettheano().graph.utils.MetaType
    else:
        assert False


######################
# Graph manipulation

def clone(output, replace=None, *args, **kwargs):
    """
    Use as theano.clone().
    TODO: Something useful with non-symbolic output ?
    """
    if not core.is_theano_object(output):
        raise ValueError("`shim.graph.clone()` is undefined for non-symbolic outputs")
    return core.gettheano().clone(output, replace, *args, **kwargs)

class ShimmedGraphExpression:
    pass

# Wrapping in a function prevents defining the class before theano is loaded
# NOTE: TheanoGraphExpression is an accumulation of patches. I'm still not 100%
# sure of its necessity (although it at least seemed the most expedient solution
# when it was created) and it is still not fully functional (in particular
# when it comes to cloning).
def get_TheanoGraphExpression():
    if get_TheanoGraphExpression.GE is None:
        from theano_shim.graph_theano import GraphExpression
        get_TheanoGraphExpression.GE = GraphExpression
    return get_TheanoGraphExpression.GE
get_TheanoGraphExpression.GE = None

# def replace_expr(expr):
#     type = expr.type
#     owner = expr.owner
#     index = expr.index
#     if name is None:
#         name = expr.name
#     # Point the parent graph to the new graph node
#     owner.outputs = [self if o is expr else o for o in owner.outputs]
#     return GraphExpression(type, owner, index, name)


#####################
# Graph compilation

def compile(inputs, outputs, *args, mode=None, **kwargs):
    """
    Use as theano.function().
    TODO: Something useful with non-symbolic output ?
    
    Parameters
    ----------
    ...
    mode: In addition to the values accepted by `theano.function`, also accepts
       a string to make it easier to use `NanGuardMode`.
       If a string, a `NanGuardMode` object is created; the string should contain
       comma separated values indicating against which values we want to guard.
       For example, with the string ``"nan,inf"``, a `NanGuardMode` object is
       created with the options ``NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)``.
    """
    if not any(core.is_theano_object(arg)
               for arg in itertools.chain([inputs, outputs], args, kwargs.values())):
        raise ValueError("`shim.graph.function()` is undefined for non-symbolic outputs")
    if mode:
        from theano.compile.nanguardmode import NanGuardMode
        if isinstance(mode, NanGuardMode):
            kwargs['mode'] = mode
        elif isinstance(mode, str):
            nanguard = 'nan' in mode
            infguard = 'inf' in mode
            bigguard = 'big' in mode
            kwargs['mode'] = NanGuardMode(nan_is_error=nanguard, inf_is_error=infguard, big_is_error=bigguard)
    # Replace dict by OrderedDict to silence Theano warnings – since 3.7, dicts
    # now have guaranteed order
    if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
        args = tuple(collections.OrderedDict(a) if type(a) is dict else a
                     for a in args)
        kwargs = {k: collections.OrderedDict(v) if type(v) is dict else v
                  for k, v in kwargs.items()}
    return core.theano.function(inputs, outputs, *args, **kwargs)

def _recursive_as_variable(exprs):
    """
    Apply `asvariable` to a set of expressions. Recurses into:

    - list
    - tuple
    - dict
    """
    if isinstance(exprs, list):
        return [_recursive_as_variable(e) for e in exprs]
    elif isinstance(exprs, tuple):
        return tuple(_recursive_as_variable(e) for e in exprs)
    elif isinstance(exprs, dict):
        return {k: _recursive_as_variable(e) for k,e in exprs.items()}
    else:
        return core.asvariable(exprs)

def eval(expr, givens=None, max_cost=20, if_too_costly='raise'):
    """
    Obtain a numerical value by evaluating an expression's graph.

    If expr is shared (or shimmed shared) variable, its `get_value()` method is
    called and the result returned.
    Otherwise, a function is compiled and evaluated. `givens` can be used
    to replace symbolic and shared variables during compilation.

    If it's not a symbolic expression at all, `expr` is simply returned.

    .. Hint:: Graph compilations can quickly become time-consuming, and one
       should avoid using `eval` too liberally, as the compilation times can
       add up.

       The default value of ``20`` is meant for negligible cost evaluations
       – things like retrieving a shared value, or evaluating an arithmetic
       expression involving constants. Thus calling `eval` without extra
       arguments amounts to a (relatively) cheap "remove the symbolic container"
       call and can be done fairly often needed. Even in this case though, 
       there remains some measurable overhead associated with the compilation
       and compile cache lookup.

       For sanity checks, it probably doesn't make sense to spend too much
       time computing a graph that will never be used in an actual computation.
       The best approach is usually to keep `max_cost` low (30 may be a
       reasonable value) and silence the ``TooCostly`` exception::

           shim.eval(my_test, max_cost=30, if_too_costly='ignore')
           
       or::
       
           try:
               shim.eval(my_test, max_cost=30)
           except shim.graph.TooCostly:
               pass

       For code which should always be executed, one can either estimate
       the expected ``max_cost``, or set it to ``None`` to disable cost
       checking entirely.

    Parameters
    ----------
    expr: Theano graph | slice | iterable
        The theano expression we want to evaluate. If a slice, each component
        of the slice ('start', 'stop', 'step') is evaluated.
        TODO: iterable
    givens: dict (optional; default: {})
        Dictionary of inputs to be past to `eval`.
    max_cost: int (optional; default: 20)
        Maximum allowable 'cost' of the expression: if `expr` is estimated to
        be more costly than this, an error is raised instead. The value `None`
        deactivates this check.
        Cost is estimated by a proxy, specifically the number of ancestors to
        `expr` in the Theano graph. By default, only expressions with 20
        ancestors or less are evaluated.
    if_too_costly: 'ignore'  |  'raise' | 'warn'
        What to do if an expression is too costly to compute.
        'ignore': do nothing, return symbolic expression.
        'raise' : (default) raise a `~shim.graph.TooCostly` exception.
        'warn'  : always evaluate (so functionally equivalent to `max_cost=None`)
                  but print a warning if the max cost is exceeded.

    Returns
    -------
    Pure Python variable of same type as `expr`.
    """
    # Bypassing code paths
    if core.isshared(expr):
        return expr.get_value()
    elif not core.is_graph_object(expr):
        return expr
    elif isinstance(expr, slice):
        kwargs = dict(givens=givens, max_cost=max_cost, if_too_costly=if_too_costly)
        return slice(eval(expr.start, **kwargs), eval(expr.stop, **kwargs),
                     eval(expr.step, **kwargs))
    # TODO: If iterable of shared vars, should just call `get_value()`

    # "Standard" code path
    if isinstance(expr, Iterable) and not isinstance(expr, cf.TerminatingTypes):
        assert all(isinstance(e, cf.TerminatingTypes) for e in expr)
        expr = _recursive_as_variable(expr)
        # scalar_expr = False
        list_of_exprs = [e for e in core._expand_args(expr) if core.is_symbolic(e)]
    else:
        list_of_exprs = [expr]
        # scalar_expr = True
    cost = sum(1 for x in ancestors(list_of_exprs)) / len(list_of_exprs)
       # `sum(1 for…` is a way of computing len with a generator
       # We take the mean because if a user passes a list, they
       # expect computation to scale with the number of terms
    # if scalar_expr:
    #     expr = expr[0]
    if max_cost is not None and cost > max_cost:
        if if_too_costly == 'raise':
            raise TooCostly(f"Expression has {cost} ancestors, which exceeds "
                            f"the limit of {max_cost}.")
        elif if_too_costly == 'warn':
            logger.warn(f"Expression has {cost} ancestors, which exceeds the "
                        f"limit of {max_cost}. Evaluating anyway because "
                        "`if_too_costly` is set to 'warn'.")
            # Only branch which allows continuing to the end of the function
        elif if_too_costly == 'ignore':
            return expr
        else:
            raise ValueError("`if_too_costly` should be either 'raise', 'warn' "
                             f"or 'ignore'. Received '{if_too_costly}'.")
    if givens is None: givens = {}
    for k, v in givens.items():
        givens[k] = core.cast(v, k.dtype)
    try:
        f = core.gettheano().function(
            [], expr, givens=givens, on_unused_input='ignore')
    except MissingInputError:
        # Make the Theano error message more friendly and useful
        symbinputs = (set(pure_symbolic_inputs(list_of_exprs)) - set(givens))
        raise MissingInputError(
            "You called `eval()` on a graph with pure symbolic inputs "
            "(i.e. non-shared symbolic inputs). Provide values for these with "
            f"the `givens` parameter.\nProblematic inputs: {symbinputs}.")
    return f()

######################
# Graph inspection

# _stablehexdigest and _tobytes also in `mackelab.utils`
def _stablehexdigest(o):
    """
    Builtin `hash` is not stable across sessions for security reasons.
    """
    import hashlib
    return hashlib.sha1(_tobytes(o)).hexdigest()

def _tobytes(o):
    if isinstance(o, Iterable) and not isinstance(o, cf.TerminatingTypes):
        return b''.join(_tobytes(oi) for oi in o)
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, bytes):
        return o
    else:
        return bytes(o)

def hash(graph):
    """
    Return a value which is consistent on equivalent graphs.

    ..Note: The implementation may change. Hashes can be used for memoization,
    but should not be used for long term storage.
    FIXME: Current implementation works by hashing the result of `shim.pprint`.
    This is not ideal, because using the same name for different variables
    might not be resolved, and shape/dtype is not included.

    Parameters
    ----------
    graph: symbolic | list of symbolics
        If a list, a single value is returned, wich is dependent on the order.
        Nested lists also work, but note that `hash([a, b]) != hash([a, [b]])`.
    """

    if (isinstance(graph, collections.Iterable)
        and not isinstance(graph, cf.TerminatingTypes)):
        return _stablehexdigest(tuple(hash(g) for g in graph))
    else:
        return _stablehexdigest(core.pprint(graph))

def is_computable(varlist, with_inputs=None):
    """
    Returns True if the variables in varlist can be numerically evaluated
    using only the inputs in `with_inputs`. In other words, the computational
    graph associated to varlist is composed only of constants and shared variables,
    along with the symbolic variables in `with_inputs`.
    If varlist is not a Theano graph, it is always computable.

    Parameters
    ----------
    varlist: expression | list of expressions
        If `varlist` is anything else than an `list`, `tuple` or `set`, it is
        wrapped with a list.
    """
    if not isinstance(varlist, (list, tuple, set)):
         # or isinstance(varlist, cf.GraphTypes)):
        # raise ValueError("theano_shim.graph.is_computable requires a list as first argument.")
        # `is_computable` requires a list as first argument.
        varlist = [varlist]
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
            # if core.is_theano_variable( set(core.gettheano().graph.basic.graph_inputs([var])).difference(with_inputs) ):
            if core.is_theano_variable( set(pure_symbolic_inputs([var])).difference(with_inputs) ):
                computable = False
                break
    return computable
    
def ancestors(varlist: Union['Variable',Iterable['Variable']],
              blockers: Optional[Collection['Variable']]=None) \
-> Generator:
    """
    Wrapper for `theano.graph.basic.ancestors`.
    Returns a generator listing the inputs to the variables in `varlist`.
    Accepts also bare variables, nested lists and dictionaries; inputs will be
    combined into one list.
    """
    varlist = [x for x in core._expand_args(varlist) if core.is_theano_object(x)]
    return core._gettheano().graph.basic.ancestors(varlist, blockers=blockers)

def graph_inputs(varlist, *args, **kwargs):
    """
    Wrapper for `theano.graph.basic.graph_inputs`.
    Returns an empty list for non symbolic variables
    Accepts also nested lists and dictionaries; inputs will be combined into
    one list.
    """
    varlist = [x for x in core._expand_args(varlist) if core.is_theano_object(x)]
    return core._gettheano().graph.basic.graph_inputs(varlist, *args, **kwargs)
inputs = graph_inputs  # Old Theano name; kept for BC with sinn

def symbolic_inputs(varlist, *args, **kwargs):
    return [v for v in graph_inputs(varlist, *args, **kwargs) if core.is_symbolic(v)]

def shared_inputs(varlist, *args, **kwargs):
    return [v for v in graph_inputs(varlist, *args, **kwargs)
              if isinstance(v, cf.SymbolicSharedType)]

def pure_symbolic_inputs(varlist, *args, **kwargs):
    return [v for v in graph_inputs(varlist, *args, **kwargs) if core.is_pure_symbolic(v)]

def vars_between(i, o=None):
    """
    Wrapper for theano.graph.variables.
    Returns an empty list for non symbolic variables
    If given only one argument, returns all variables it depends on.
    """
    if o is None:
        o = i
        i = graph_inputs(o)
    if isinstance(i, cf.GraphTypes):
        i = [i]
    if isinstance(o, cf.GraphTypes):
        o = [o]
    if ( not isinstance(i, collections.Iterable)
         or isinstance(i, str) ):
        raise ValueError(
            "Arguments to theano_shim.graph.vars_between must be lists.")
    if core.is_theano_object(i, o):
        return core._gettheano().graph.basic.vars_between(i, o)
    else:
        return []
variables = vars_between  # Old Theano name; kept for BC with sinn

def is_same_graph(var1, var2, givens=None):
    """
    Wraps theano.graph.toolbox.is_same_graph().
    Returns True if `var1` and `var2` perform the same computation.
    If either `var1` or `var2` is not a graph object, returns the result of
    `var1 == var2`.
    """
    if not (core.is_theano_object(var1) and core.is_theano_object(var2)):
        return var1 == var2
    else:
        return core._gettheano().graph.toolbox.is_same_graph(var1, var2, givens)
