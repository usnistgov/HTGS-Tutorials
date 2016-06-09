# HTGS-Tutorials

In this repository are the tutorials for the [HTGS API](https://github.com/usnistgov/htgs).

## [**tutorial1**](https://pages.nist.gov/HTGS/doxygen/tutorial1.html)
Adds two numbers together, purely for demonstration purposes of HTGS. (Note: Computational tasks for an algorithm's implementation should provide enough computational complexity to justify shipping data between tasks, see [Figure](https://pages.nist.gov/HTGS/doxygen/figures/blocksize-impact.png))

## [**tutorial2**](https://pages.nist.gov/HTGS/doxygen/tutorial2.html)
Computes the Hadamard product. Provided are two versions:

1. hadamard-product - writes blocks of matrices to disk and processes them overlapping I/O with the Hadamard product computation
2. hadamard-produce-no-read - generates blocks of matrices (no disk reading) and computes the Hadamard product

