//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_MATRIXFACTORCUDADATA_H
#define HTGS_TUTORIALS_MATRIXFACTORCUDADATA_H

#include <htgs/api/IData.hpp>

#include "MatrixBlockData.h"
template <class T>
class MatrixFactorCudaData : public htgs::IData
{
 public:

  MatrixFactorCudaData(const std::shared_ptr<MatrixBlockData<std::shared_ptr<htgs::MemoryData<T>>>> &triMatrix,
                   const std::shared_ptr<MatrixBlockData<T>> &unfactoredMatrix)
      : unfactoredMatrix(unfactoredMatrix), triMatrix(triMatrix) {}

  const std::shared_ptr<MatrixBlockData<T>> &getUnFactoredMatrix() const {
    return unfactoredMatrix;
  }
  const std::shared_ptr<MatrixBlockData<std::shared_ptr<htgs::MemoryData<T>>>> &getTriangleMatrix() const {
    return triMatrix;
  }

 private:
  std::shared_ptr<MatrixBlockData<T>> unfactoredMatrix;
  std::shared_ptr<MatrixBlockData<std::shared_ptr<htgs::MemoryData<T>>>> triMatrix;
};

#endif //HTGS_TUTORIALS_MATRIXFACTORCUDADATA_H
