//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_MATRIXFACTORDATA_H
#define HTGS_TUTORIALS_MATRIXFACTORDATA_H

#include <htgs/api/IData.hpp>

#include "MatrixBlockData.h"
#include "../types/MatrixTypes.h"

class MatrixFactorData : public htgs::IData
{
 public:

  MatrixFactorData(const std::shared_ptr<MatrixBlockData<data_ptr>> &triMatrix,
                   const std::shared_ptr<MatrixBlockData<data_ptr>> &unfactoredMatrix)
      : unfactoredMatrix(unfactoredMatrix), triMatrix(triMatrix) {}

  const std::shared_ptr<MatrixBlockData<data_ptr>> &getUnFactoredMatrix() const {
    return unfactoredMatrix;
  }
  const std::shared_ptr<MatrixBlockData<data_ptr>> &getTriangleMatrix() const {
    return triMatrix;
  }

 private:
  std::shared_ptr<MatrixBlockData<data_ptr>> unfactoredMatrix;
  std::shared_ptr<MatrixBlockData<data_ptr>> triMatrix;
};

#endif //HTGS_TUTORIALS_MATRIXFACTORDATA_H
