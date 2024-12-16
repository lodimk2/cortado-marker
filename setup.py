from setuptools import setup

setup(
    name="cortado-marker",  # Updated package name
    version="0.1.0",
    author="Musaddiq Lodi",
    author_email="lodimk2@vcu.edu",
    description="CORTADO: hill Climbing Optimization foR cell-Type specific mArker gene DiscOvery",
    install_requires=[
        "numpy>=1.20.0",  # Added version constraints for better dependency management
        "matplotlib>=3.4.0",
        "scipy>=1.6.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "scanpy>=1.9.1",  # Added scanpy with a version known for compatibility
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cortado-marker",  # Updated to new repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Updated to a more recent Python version (if applicable)
)
