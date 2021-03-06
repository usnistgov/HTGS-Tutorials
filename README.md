# HTGS-Tutorials

In this repository we present the tutorials for the [HTGS API](https://github.com/usnistgov/htgs).

## Getting Started
Please refer to [**tutorial0**](https://pages.nist.gov/HTGS/doxygen/tutorial0.html) for details on how to compile and run
the tutorials. 

### Dependencies
[HTGS](https://github.com/usnistgov/htgs) is required, location is specified using:
cmake -DHTGS_INCLUDE_DIR=<dir>

(Optional) [OpenBlas](http://www.openblas.net/); required for [Tutorial3b](https://pages.nist.gov/HTGS/doxygen/tutorial3b.html)

## [**tutorial1**](https://pages.nist.gov/HTGS/doxygen/tutorial1.html)
Adds two numbers together, purely for demonstration purposes of HTGS. (Note: Computational tasks for an algorithm's implementation should provide enough computational complexity to justify shipping data between tasks, see [Figure](https://pages.nist.gov/HTGS/doxygen/figures/blocksize-impact.png))

## [**tutorial2**](https://pages.nist.gov/HTGS/doxygen/tutorial2.html)
Computes the Hadamard product. The tutorial uses the tutorial-utils to read/write blocks of matrices. Provided are two versions:

1. hadamard-product - reads blocks of matrices from disk and computes the Hadamard product
2. hadamard-product-no-read - generates blocks of matrices (no disk reading) and computes the Hadamard product


## [**tutorial3**](https://pages.nist.gov/HTGS/doxygen/tutorial3a.html)
Computes matrix multiplication. Re-uses data representations found in tutorial2, and implements matrix multiplication in HTGS.
