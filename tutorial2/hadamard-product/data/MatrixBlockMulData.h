//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXBLOCKMULDATA_H
#define HTGS_MATRIXBLOCKMULDATA_H

#include <htgs/api/IData.hpp>

class MatrixBlockMulData : public htgs::IData {
 public:

  MatrixBlockMulData(const std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> &matrixA, const std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> &matrixB) :
  matrixA(matrixA), matrixB(matrixB) { }

  const std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> &getMatrixA() const {
    return matrixA;
  }
  const std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> &getMatrixB() const {
    return matrixB;
  }

 private:
  std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> matrixA;
  std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> matrixB;
};

#endif //HTGS_MATRIXBLOCKMULDATA_H
