# HTGS-Tutorials

In this repository we present the tutorials for the [HTGS API](https://github.com/usnistgov/htgs).

## [**tutorial1**](https://pages.nist.gov/HTGS/doxygen/tutorial1.html)
Adds two numbers together, purely for demonstration purposes of HTGS. (Note: Computational tasks for an algorithm's implementation should provide enough computational complexity to justify shipping data between tasks, see [Figure](https://pages.nist.gov/HTGS/doxygen/figures/blocksize-impact.png))

## [**tutorial2**](https://pages.nist.gov/HTGS/doxygen/tutorial2.html)
Computes the Hadamard product. The tutorial uses the tutorial-utils to read/write blocks of matrices. Provided are two versions:

1. hadamard-product - reads blocks of matrices from disk and computes the Hadamard product
2. hadamard-product-no-read - generates blocks of matrices (no disk reading) and computes the Hadamard product

