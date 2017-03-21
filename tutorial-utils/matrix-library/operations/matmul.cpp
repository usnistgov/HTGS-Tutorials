//
// Created by tjb3 on 3/21/17.
//



#include <cstddef>
#include "../../util-matrix.h"


void computeMatMul(size_t M, size_t N, size_t K, double alpha, double *A, size_t LDA, double *B, size_t LDB, double beta, double *C, size_t LDC, bool ColMajor)
{
#ifdef USE_CUDA
// TODO: Add CUDA
#elif USE_OPENBLAS
  cblas_dgem(ColMajor ? CblasColMajor : CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
#else
  if (ColMajor) {
    for (size_t aRow = 0; aRow < M; aRow++) {
      for (size_t bCol = 0; bCol < N; bCol++) {
        double sum = 0.0;
        for (size_t k = 0; k < K; k++) {
          sum += alpha * A[IDX2C(aRow, k, LDA)] * B[IDX2C(k, bCol, LDB)] + (beta != 0.0 ? C[IDX2C(aRow, bCol, LDC)] * beta : 0.0);
        }
        C[IDX2C(aRow, bCol, LDC)] = sum;
      }
    }
  }
  else
  {
    for (size_t aRow = 0; aRow < M; aRow++) {
      for (size_t bCol = 0; bCol < N; bCol++) {
        double sum = 0.0;
        if (beta != 0.0)
          sum = beta * C[aRow * LDC + bCol];
        for (size_t k = 0; k < K; k++) {
          sum += alpha * A[aRow * LDA + k] * B[k * LDB + bCol] + (beta != 0.0 ? C[aRow *LDC + bCol] * beta : 0.0);
        }
        C[aRow * LDC + bCol] = sum;
      }
    }
  }

#endif
}


bool validateMatMulResults(size_t numItemsPrint, double *a1, double *a2, size_t size)
{
  size_t count = 0;
  for (size_t i = 0; i < size; i++)
  {
    if (a1[i] != a2[i])
    {
      std::cout << i << ": a1 = " << a1[i] << " a2 = " << a2[i] << std::endl;
      if (count == numItemsPrint) {
        return false;
      }
      count++;
    }
  }

  return true;
}
