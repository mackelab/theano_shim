# This module is only imported when Theano is loaded, and therefor can import
# and use theano in the global namespace without issue

from typing import Type
import theano
from theano.tensor import DimShuffle, Apply, TensorType, Variable
from .config import config as cf

class ElemwiseWrapper(theano.tensor.Elemwise):
    def __init__(self, VarType: Type[Variable], *args, **kwargs):
        self.VarType = VarType
        super().__init__(*args, **kwargs)
        
    def make_node(self, *inputs):
        """
        (Copied from theano.tensor.Elemwise to change one line)
        
        If the inputs have different number of dimensions, their shape
        is left-completed to the greatest number of dimensions with 1s
        using DimShuffle.
        """
        inputs = list(map(theano.tensor.basic.as_tensor_variable, inputs))
        out_dtypes, out_broadcastables, inputs = self.get_output_info(
            DimShuffle, *inputs
        )
        outputs = [
            self.VarType(TensorType(dtype=dtype, broadcastable=broadcastable))  # <-- This line changed
            for dtype, broadcastable in zip(out_dtypes, out_broadcastables)
        ]
        return Apply(self, inputs, outputs)
        
class GraphExpression(Variable,
                      theano.tensor._tensor_py_operators):
    """
    Mixin class to create a node as an computational graph representing
    a variable or expression. (As opposed to an operation.)

    The initialization signature matches that of a graph node, allowing
    internal graph functions such as `copy()` and `clone()` to work.
    When mixing into another class, make sure this initialization signature
    remains valid (reminder: multiple inheritance precedence goes left to right).

    In addition to the default initialization, it is also possible to
    pass an already instantiated tensor (which we refer to as `expr`).
    This can be useful to wrap a plain tensor with a custom symbolic
    subclass. In this case, one of two things happen
    - If `expr` is already of the type that would be returned, return
      it unchanged.
    - Otherwise, add an identity operation whose output is of the
      new type.
    """
    def __new__(cls, expr_or_type, owner=None, index=None, name=None):
        if isinstance(expr_or_type, cls):
            # We already have an expression of the target type
            expr_or_type._skip_init = True
            return expr_or_type
        elif isinstance(expr_or_type, cf.SymbolicExpressionType):
            # We have an expression, but not of the target type
            # Create a new child node – identity op with correct type
            if owner is not None or index is not None:
                raise TypeError("`owner` and `index` should only be "
                                "specified when `expr_or_type` is a "
                                "tensor type.")
            new_expr = ElemwiseWrapper(
                cls,
                theano.scalar.basic.identity,
                name=f"Elemwise{{identity (-> {cls})}}"
                )(expr_or_type)
            if name:
                new_expr.name = name
            return new_expr
        else:
            # We have a type. Just follow the standard Variable init
            assert super().__new__ is object.__new__
            return super().__new__(cls)
    def __init__(self, expr_or_type, owner=None, index=None, name=None):
        if getattr(self, '_skip_init', False):
            del self._skip_init
            return
        elif isinstance(expr_or_type, theano.tensor.TensorType):
            super().__init__(expr_or_type, owner, index, name)
        
