# Theano/Numpy shim

Copyright (c) 2017-2020 Alexandre René

## Description
A simple convenient exchangeable interface, to avoid the need of
conditionals just to select between e.g. `T.sum` and `np.sum`.
For cases where conditionals are needed, helper functions are provided.

Beyond the functions common to the `numpy` and `theano.tensor` namespaces,
this module also provides an interchangeable interfaces to common operations
such as type casting and checking, assertions and rounding, as well
as 'shim' datatypes for random number streams and shared variables.

## Usage

### Loading the shim
At the top of your code, include the line

    import theano_shim as shim

By default this will not even try to load Theano, so you can use it on
a machine where Theano is not installed.
To 'switch on' Theano, add the following below the import:

    shim.load_theano()

You can switch it back to using pure Numpy `shim.load(False)`. Theano can also be loaded with `shim.load()`

### Writing your own conditionals
In some cases, you may find that the functions provided here are
insufficiently flexible for your needs, and decide to treat the
Theano and Numpy cases yourself. `theano_shim` provides a few functions
to aid this task:

- `is_theano_variable` checks whether a variable is a symbolic variable
- `is_theano_object` checks whether a variable is any Theano object
   (including shared variables)
- `isshared` checks whether a variable is a shared object (either a Theano
   shared variable, or an instance `ShimmedTensorShared` which this module provides).

You can also test on `theano_shim.config.use_theano` to check whether Theano
is loaded.

## Development status
This is an alpha release, so you should not blindly trust
the functions in this package.
More importantly, the functions provide far from complete
coverage of the Theano/Numpy APIs, so expect to need
to extend it for your needs (don't forget to share your
contributions back upstream ! =) ). For this reason, a
'Development Installation' is the preferred means of
installing this module, so it doesn't need to be reinstalled
after every modification. This can done by changing to the
directory containing this file and executing

    pip install -e .
