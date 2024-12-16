"""
Cortado package
================

A brief description of the package.
"""

__version__ = '0.1.22'
__author__ = 'Musaddiq K Lodi'
__license__ = 'MIT'

# Import key functions and classes
from .data import load_data
from .marker_genes import *  # Import any specific functions you need
from .hill_climbing import *  # Import any specific functions you need
from .evaluate import *  # Import any specific functions you need
from .utils import *  # Import any specific functions you need

# Optional: initialization code
def init():
    # Code to be executed when the package is imported
    print("Cortado package initialized!")

init()
