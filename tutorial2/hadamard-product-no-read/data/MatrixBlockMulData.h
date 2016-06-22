//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXBLOCKMULDATA_H
#define HTGS_MATRIXBLOCKMULDATA_H

class MatrixBlockMulData : public htgs::IData {
 public:

  MatrixBlockMulData(const std::shared_ptr<MatrixBlockData<double *>> &matrixA,
                     const std::shared_ptr<MatrixBlockData<double *>> &matrixB) :
      matrixA(matrixA), matrixB(matrixB) {}

  const std::shared_ptr<MatrixBlockData<double *>> &getMatrixA() const {
    return matrixA;
  }
  const std::shared_ptr<MatrixBlockData<double *>> &getMatrixB() const {
    return matrixB;
  }

 private:
  std::shared_ptr<MatrixBlockData<double *>> matrixA;
  std::shared_ptr<MatrixBlockData<double *>> matrixB;
};

#endif //HTGS_MATRIXBLOCKMULDATA_H
