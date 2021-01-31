
from setuptools import setup

setup(
      name='theano_shim',
      version='0.3.0dev',
      description="A simple interface to easily switch between using Theano and pure Numpy",

      author="Alexandre René",
      author_email="alexandre.rene@freeneurons.ca",

      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only'
      ],

      packages=["theano_shim"],

      install_requires=['numpy', 'scipy', 'theano-pymc']
     )
