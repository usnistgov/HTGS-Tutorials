//
// Created by tjb3 on 3/21/17.
//

#ifndef HTGS_TUTORIALS_MATMUL_H
#define HTGS_TUTORIALS_MATMUL_H

#include <cstddef>
void initMatMul(size_t numThreads);

void computeMatMul(size_t M, size_t N, size_t K, double alpha, double *A, size_t LDA, double *B, size_t LDB, double beta, double *C, size_t LDC, bool ColMajor);

bool validateMatMulResults(size_t numItemsPrint, double *a1, double *a2, size_t size);

#endif //HTGS_TUTORIALS_MATMUL_H
