//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_MATRIXFACTORCUDADATA_H
#define HTGS_TUTORIALS_MATRIXFACTORCUDADATA_H

#include <htgs/api/IData.hpp>

#include "MatrixBlockData.h"

class MatrixFactorCudaData : public htgs::IData
{
 public:

  MatrixFactorCudaData(const std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> &triMatrix,
                   const std::shared_ptr<MatrixBlockData<data_ptr>> &unfactoredMatrix)
      : unfactoredMatrix(unfactoredMatrix), triMatrix(triMatrix) {}

  const std::shared_ptr<MatrixBlockData<data_ptr>> &getUnFactoredMatrix() const {
    return unfactoredMatrix;
  }
  const std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> &getTriangleMatrix() const {
    return triMatrix;
  }

 private:
  std::shared_ptr<MatrixBlockData<data_ptr>> unfactoredMatrix;
  std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> triMatrix;
};

#endif //HTGS_TUTORIALS_MATRIXFACTORCUDADATA_H
