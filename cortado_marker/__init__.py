"""
Cortado package
================

A brief description of the package.
"""

__version__ = '0.1.2'
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
    print("Thank you for using CORTADO: Hill Climbing Optimization for Cell-Type Specific Marker Gene Discovery. For questions, comments or concerns, please reach out to Musaddiq Lodi @ lodimk2@vcu.edu.")

init()
