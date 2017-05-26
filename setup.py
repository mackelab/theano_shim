
from setuptools import setup

setup(
      name='theano_shim',
      version='0.2.2',
      description="A simple interface to easily switch between using Theano and pure Numpy",

      author="Alexandre René",
      author_email="alexandre.rene@caesar.de",

      license='MIT',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only'
      ],

      packages=["theano_shim"],

      install_requires=['numpy', 'scipy']
     )
