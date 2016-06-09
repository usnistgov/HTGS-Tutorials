//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXBLOCKMULDATA_H
#define HTGS_MATRIXBLOCKMULDATA_H

#include <htgs/api/IData.hpp>

template<class T>
class MatrixBlockMulData : public htgs::IData {
 public:

    MatrixBlockMulData(const std::shared_ptr<MatrixBlockData<T>> &matrixA,
                       const std::shared_ptr<MatrixBlockData<T>> &matrixB) :
  matrixA(matrixA), matrixB(matrixB) { }

    const std::shared_ptr<MatrixBlockData<T>> &getMatrixA() const {
    return matrixA;
  }

    const std::shared_ptr<MatrixBlockData<T>> &getMatrixB() const {
    return matrixB;
  }

 private:
    std::shared_ptr<MatrixBlockData<T>> matrixA;
    std::shared_ptr<MatrixBlockData<T>> matrixB;
};

#endif //HTGS_MATRIXBLOCKMULDATA_H
