"""
Shimmed Graph utilities:
   graph compilation, traversal, listing inputs...
"""
import collections
from collections.abc import Iterable
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
    if cf.use_theano:
        import theano.gof as gof
        MissingInputError = gof.MissingInputError
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
        GraphExpressionMeta = core.gettheano().gof.utils.MetaObject
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
def get_TheanoGraphExpression():
    if get_TheanoGraphExpression.TGE is None:
        theano = core.gettheano()
        class TheanoGraphExpression(theano.gof.graph.Variable,
                                    theano.tensor._tensor_py_operators):
            """
            Mixin class to create a node a an computational graph representing
            a variable or expression. (As opposed to an operation.)

            The initialization signature matches that of a graph node, allowing
            internal graph functions such as `copy()` to work.
            When mixing into another class, make sure this initialization signature
            remains valid (reminder: multiple inheritance precedence goes left to right).

            In addition to the default initialization,
            """
            def __init__(self, expr_or_type, owner=None, index=None, name=None):
                if isinstance(expr_or_type, cf.SymbolicExpressionType):
                    # We actually just passed an expression
                    # Add a dummy entry to the graph, so we don't invalidate an
                    # existing variable
                    expr = expr_or_type.copy()
                        # .copy() creates a new node with for the 'identity'
                        # operation and `expr_or_type` as only input.
                        # This allows us to modify `expr` without
                        # invalidating the variable that was passed as argument.
                    type = expr.type
                    owner = expr.owner
                    index = expr.index
                    if name is None:
                        name = expr.name
                    # Point the parent graph to the new graph node
                    if owner is not None:
                        owner.outputs = [self if o is expr else o
                                         for o in owner.outputs]
                else:
                    type = expr_or_type
                super().__init__(type, owner, index, name)
        get_TheanoGraphExpression.TGE = TheanoGraphExpression
    return get_TheanoGraphExpression.TGE
get_TheanoGraphExpression.TGE = None

def replace_expr(expr):
    type = expr.type
    owner = expr.owner
    index = expr.index
    if name is None:
        name = expr.name
    # Point the parent graph to the new graph node
    owner.outputs = [self if o is expr else o for o in owner.outputs]
    return GraphExpression(type, owner, index, name)



#####################
# Graph compilation

def compile(inputs, outputs, *args, **kwargs):
    """
    Use as theano.function().
    TODO: Something useful with non-symbolic output ?
    """
    if not any(core.is_theano_object(arg)
               for arg in itertools.chain([inputs, outputs], args, kwargs.values())):
        raise ValueError("`shim.graph.function()` is undefined for non-symbolic outputs")
    return core.theano.function(inputs, outputs, *args, **kwargs)

def eval(expr, givens=None, max_cost=10, if_too_costly='raise'):
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

       The default value of ``10`` is meant for negligible cost evaluations
       – things like retrieving a shared value, or evaluating an arithmetic
       expression involving constants. Thus calling `eval` without extra
       arguments amounts to a cheap "remove the symbolic container" call and
       can be done whenever needed.

       For sanity checks, it probably doesn't make sense to spend too much
       time computing a graph that will never be used in an actual computation.
       The best approach is usually to keep `max_cost` low (30 may be a
       reasonable value) and silence the ``TooCostly`` exception::

           try:
               shim.eval(my_test, max_cost=30)
           except shim.graph.TooCostly:
               pass

       For code which should always be executed, one can either estimate
       the expected ``max_cost``, or set it to ``None`` to disable cost
       checking entirely.

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
        expr = [core.asvariable(e) for e in expr]
        expr_list = True
    else:
        expr = [expr]
        expr_list = False
    cost = len(core.gettheano().gof.graph.ancestors(expr)) / len(expr)
       # We take the mean because if a user passes a list, they
       # expect computation to scale with the number of terms
    if not expr_list:
        expr = expr[0]
    if max_cost is not None and cost > max_cost:
        if if_too_costly == 'raise':
            raise TooCostly("Expression has {} ancestors, which exceeds the "
                            "limit of {}.".format(cost, max_cost))
        else:
            return expr
    else:
        if givens is None: givens = {}
        for k, v in givens.items():
            givens[k] = core.cast(v, k.dtype)
        try:
            f = core.gettheano().function(
                [], expr, givens=givens, on_unused_input='ignore')
        except MissingInputError:
            # Make the Theano error message more friendly and useful
            symbinputs = (set(shim.graph.pure_symbolic_inputs(expr))
                          - set(givens))
            raise MissingInputError(
                "You called `eval()` on a graph with pure symbolic inputs "
                "(shared variables are fine) Provide values for these with "
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
            # if core.is_theano_variable( set(core.gettheano().gof.graph.inputs([var])).difference(with_inputs) ):
            if core.is_theano_variable( set(pure_symbolic_inputs([var])).difference(with_inputs) ):
                computable = False
                break
    return computable

def inputs(varlist, *args, **kwargs):
    """
    Wrapper for theano.gof.graph.inputs.
    Returns an empty list for non symbolic variables
    """
    if isinstance(varlist, cf.GraphTypes):
        # Wrap calls on single variables with a list
        varlist = [varlist]
    elif ( not isinstance(varlist, collections.Iterable)
         or isinstance(varlist, str) ):
        raise ValueError("theano_shim.graph.inputs requires a list as first argument.")
    if core.is_theano_object(varlist):
        return core._gettheano().gof.graph.inputs(varlist, *args, **kwargs)
    else:
        return []

def symbolic_inputs(varlist, *args, **kwargs):
    return [v for v in inputs(varlist, *args, **kwargs) if core.is_symbolic(v)]

def shared_inputs(varlist, *args, **kwargs):
    return [v for v in inputs(varlist, *args, **kwargs)
              if isinstance(v, cf.SymbolicSharedType)]

def pure_symbolic_inputs(varlist, *args, **kwargs):
    return [v for v in inputs(varlist, *args, **kwargs) if core.is_pure_symbolic(v)]

def variables(i, o=None):
    """
    Wrapper for theano.gof.graph.variables.
    Returns an empty list for non symbolic variables
    If given only one argument, returns all variables it depends on.
    """
    if o is None:
        o = i
        i = inputs(o)
    if isinstance(i, cf.GraphTypes):
        i = [i]
    if isinstance(o, cf.GraphTypes):
        o = [o]
    if ( not isinstance(i, collections.Iterable)
         or isinstance(i, str) ):
        raise ValueError(
            "Arguments to theano_shim.graph.variables must be lists.")
    if core.is_theano_object(i, o):
        return core._gettheano().gof.graph.variables(i, o)
    else:
        return []

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
