"""
Global configuration variables
"""
import sys
from warnings import warn
from numbers import Number
# Make it safe to assume Python 3 when testing version
if sys.version_info.major < 3:
    raise RuntimeError("theano_shim requires Python 3. You are using {}."
                       .format(sys.version))
from collections import OrderedDict
from typing import Union
import numpy as np

# These forms will retrieve theano or theano.tensor, whether or not cf.use_theano == True,
# as long as any module in the stack has loaded them
# These functions should only be used within this package, for cases where we need to treat
# Theano arguments even if use_theano == False. Type-checking is an example of such a use case.
def _gettheano():
    if 'theano' in sys.modules:
        return sys.modules['theano']
    else:
        raise RuntimeError("Tried to access theano, but it has not been loaded.")
def _getT():
    if 'theano' in sys.modules:
        return sys.modules['theano'].tensor
    else:
        raise RuntimeError("Tried to access theano.tensor, but Theano has not been loaded.")

def make_32bit_var(module, var):
    """
    Modify a module's variable so that it's a single precision float.
    The original value is saved under the variable name 'var64',
    where 'var' is the provided variable name.
    """
    value = getattr(module, var)
    dname = var + '64'
    if not hasattr(module, dname):
        # Save the 64bit value before overwriting the variable
        setattr(module, dname, value)
    # Overwrite with 32bit value
    setattr(module, var, np.float32(value))

# Singleton class copied from mackelab_toolbox.utils
class Singleton(type):
    """Singleton metaclass

    Although singletons are usually an anti-pattern, I've found them useful in
    a few cases, notably for a configuration class storing dynamic attributes
    in the form of properties.

    Example
    -------
    >>> from mackelab_toolbox.utils import Singleton
    >>> import sys
    >>>
    >>> class Config(metaclass=Singleton):
    >>>     def num_modules(self):
    >>>         return len(sys.modules)
    """
    __instance = None
    def __new__(metacls, name, bases, dct):
        cls = super().__new__(metacls, name, bases, dct)
        cls.__instance = None
        cls.__new__ = metacls.__clsnew__
        return cls
    @staticmethod
    def __clsnew__(cls, *args, **kwargs):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super(cls, cls).__new__(cls,*args,**kwargs)
        return cls.__instance

class Config(metaclass=Singleton):
    inf = None
    _library = 'numpy'  # Currently can only be set to 'numpy' or 'theano'
    _library_values = ('numpy', 'theano')
    _floatX = 'float64'  # Use this if Theano is not loaded
    _shared_types = ()  # core appends `ShimmedTensorShared` to this

    @property
    def library(self):
        return self._library
    @library.setter
    def library(self, value):
        if value not in self._library_values:
            libnames = ', '.join(f"'{l}'" for l in self._library_values)
            raise ValueError("`library` attribute can only be set to one of "
                             f"{libnames}. Received '{value}'.")
        self._library = value
        return value

    # For backwards compatibility
    @property
    def use_theano(self):
        return self.library == 'theano'
    @use_theano.setter
    def use_theano(self, value):
        assert isinstance(value, bool)
        warn("Set `shim.config.library` instead of `shim.config.use_theano`.")
        if value:
            self.library = 'theano'
        else:
            self.library = 'numpy'

    # @property
    # def Numeric(self):
    #     if self.use_theano:
    #         return Union[np.ndarray, T.TensorVariable]
    #     else:
    #         return Union[np.ndarray]
    #
    symbolic_updates = OrderedDict()
        # Stores a Theano update dictionary

    RandomStreams = None

    # TerminatingTypes is a tuple of all types which may be iterable but
    # are treated as a single unit. This is used when recursively expanding
    # an argument in core._expand_args.
    # For example, if the sparse module is loaded, sparse types are added to TerminatingTypes,
    # since we will never store a Theano object as an element of a SciPy sparse matrix
    _TerminatingTypes = (bytes,str,np.ndarray)

    def add_terminating_types(self, type_list):
        self._TerminatingTypes = tuple( set(type_list).union( self._TerminatingTypes ) )
    @property
    def TerminatingTypes(self):
        if 'theano' in sys.modules:
            dyntypes = ((self.SymbolicType,) + self.GraphTypes
                        + self.ConstantTypes)
        else:
            dyntypes = self.ConstantTypes
        if 'xarray' in sys.modules:
            xr = sys.modules['xarray']
            dyntypes += (xr.Dataset, xr.DataArray)
        return self._TerminatingTypes + dyntypes

    @staticmethod
    def make_constants_32bit():
        # TODO: Have a maintainable list of these constants
        make_32bit_var(np, 'pi')

    @property
    def SymbolicType(self):
        """
        Matches:
            Tensor variables
            Expressions
            Shared variables
            Tensor constants
        Does not match:
            Random variables
        """
        if 'theano' not in sys.modules: return ()
        else: return _gettheano().gof.graph.Variable
    # For Theano, the expression type is just the generic symbolic type
    SymbolicExpressionType = SymbolicType
    @property
    def GraphTypes(self):
        """
        Matches any node in a computational graph:
            Tensor variables
            Expressions
            Shared variables
            Tensor constants
            Random variables
        """
        if 'theano' not in sys.modules: return ()
        else: return (self.SymbolicType, self.SymbolicRNGType)
    @property
    def ShimmedAndGraphTypes(self):
        """
        Matches any node in a computational graph:
            Tensor variables
            Expressions
            Shared variables
            Tensor constants
            Random variables
        + ShimmedShared variables
        """
        return self.SharedTypes + self.GraphTypes
    @property
    def PureSymbolicTypes(self):
        """
        Matches:
            Tensor variables
            Expressions
        Does not match:
            Shared variables
            Tensor constants
            Random variables

        .. Note: This does not correspond to Theano's `TensorType`. For that
           see `~config.Config.TensorDescType`.
        """
        if 'theano' not in sys.modules: return ()
        else: return (_getT().TensorVariable, _gettheano().scalar.basic.ScalarVariable)
    TensorType = PureSymbolicTypes
    @property
    def SymbolicConstantType(self):
        """
        Matches:
            Tensor constants
        """
        if 'theano' not in sys.modules: return ()
        else: return _gettheano().gof.Constant
    # @property
    # def TensorConstantType(self):
    #   """
    #   The difference between Constant and TensorConstant, is that
    #   TensorConstant adds the py_operators mixin, which define tensor
    #   operations.
    #   => I think this is a distinction best hidden from the interface.
    #   """
    #     if 'theano' not in sys.modules: return ()
    #     else: return _getT().TensorConstant
    # @property
    # def ScalarConstantType(self):
    #   """
    #   Same as above: best not to make distinction with ScalarConstants in
    #   interface. Even shim.asvariable(0) doesn't match.
    #   """
    #     if 'theano' not in sys.modules: return ()
    #     else: return _gettheano().scalar.ScalarConstant
    @property
    def SymbolicSharedType(self):
        """
        Matches only symbolic (not shimmed) shared variables.
        """
        if 'theano' not in sys.modules: return ()
        else: return _getT().sharedvar.SharedVariable
    @property
    def SharedTypes(self):
        """Return a tuple containing all *loaded* shared types.
        ShimmedTensorShared is always included, and if theano is loaded, so is
        SymbolicSharedType.
        """
        if 'theano' in sys.modules:
            return self._shared_types + (self.SymbolicSharedType,)
        else:
            return self._shared_types
    @property
    def ConstantTypes(self):
        """
        Matches both symbolic and literal constants. Specifically, returns
        ``(config.SymbolicConstantType, Number)``
        """
        if 'theano' in sys.modules:
            return (self.SymbolicConstantType, Number)
        else:
            return (Number,)
    @property
    def SymbolicRNGType(self):
        if 'theano' not in sys.modules: return ()
        else: return _getT().shared_randomstreams.RandomStreams
    @property
    def RNGTypes(self):
        if 'theano' in sys.modules:
            return (_getT().shared_randomstreams.RandomStreams,
                    np.random.Generator, np.random.RandomState)
        else:
            return (np.random.Generator, np.random.RandomState)
    @property
    def RandomStateType(self):
        if 'theano' not in sys.modules: return ()
        else: return _getT().shared_randomstreams.RandomStateSharedVariable
    @property
    def CompiledType(self):
        if 'theano' not in sys.modules: return ()
        else: return _gettheano().compile.function_module.Function
    @property
    def TensorDescType(self):
        """
        An object, which stores the information necessary to create a tensor.
        In Theano, this is the `TensorType`, which stores the dtype and
        broadcast pattern.
        """
        if 'theano' not in sys.modules: return ()
        else: return _getT().TensorType

    @property
    def floatX(self):
        if 'theano' in sys.modules:
            # Update internal floatX, in case theano's was changed
            self._floatX = _gettheano().config.floatX
            return self._floatX
        else:
            # TODO: Read theanorc ?
            return self._floatX  # plain python float

    @floatX.setter
    def floatX(self, floatX):
        if floatX not in ['float32', 'float64']:
            raise ValueError("'floatX' value must be a string, either "
                             "'float32' or 'float64'.")
        if 'theano' in sys.modules:
            _gettheano().config.floatX = floatX
            self._floatX = floatX
        else:
            self._floatX = floatX
        if floatX == 'float32':
            # Make NumPy constants 32-bit so they don't trigger
            # upcasts when used in expressions
            self.make_constants_32bit()

    @property
    def compute_test_value(self):
        if 'theano' in sys.modules:
            return _gettheano().config.copmute_test_value
        else:
            return "Theano is not loaded"

    @compute_test_value.setter
    def compute_test_value(self, flag):
        if 'theano' in sys.modules:
            _gettheano().config.compute_test_value = flag
        else:
            warn("Setting `compute_test_value` has no effect when Theano "
                 "is not loaded.")

config = Config()
