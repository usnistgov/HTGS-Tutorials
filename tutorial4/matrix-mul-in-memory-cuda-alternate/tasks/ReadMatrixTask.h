//
// Created by tjb3 on 2/23/16.
//
#ifndef HTGS_READMATRIXTASK_H
#define HTGS_READMATRIXTASK_H

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../memory/MatrixMemoryRule.h"
#include "../../../tutorial-utils/util-matrix.h"
#include "../data/MatrixRequestData.h"
#include "../data/MatrixBlockData.h"

class ReadMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<double *>>
{

 public:

  ReadMatrixTask(int numThreads, MatrixType type, int blockSize, long fullMatrixWidth, long fullMatrixHeight, double *matrix, std::string matrixName);

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data);

  virtual std::string getName() {
    return "ReadMatrixTask(" + matrixName + ")";
  }
  virtual ReadMatrixTask *copy() {
    return new ReadMatrixTask(this->getNumThreads(), this->type, blockSize, fullMatrixWidth, fullMatrixHeight, matrix, matrixName);
  }

  int getNumBlocksRows() const {
    return numBlocksRows;
  }
  int getNumBlocksCols() const {
    return numBlocksCols;
  }
 private:
  MatrixType type;
  double *matrix;
  int blockSize;
  long fullMatrixWidth;
  long fullMatrixHeight;
  int numBlocksRows;
  int numBlocksCols;
  std::string matrixName;


};

#endif //HTGS_GENERATEMATRIXTASK_H
