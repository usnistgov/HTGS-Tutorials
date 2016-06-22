//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_OUTPUTTASK_H
#define HTGS_OUTPUTTASK_H

#include <htgs/api/ITask.hpp>
#include "../data/MatrixRequestData.h"
#include "../data/MatrixBlockData.h"
#include "../../../tutorial-utils/util-matrix.h"
class OutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  OutputTask(double *matrix, long fullMatrixWidth, long fullMatrixHeight, int blockSize);

  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data);

  virtual std::string getName() {
    return "OutputTask";
  }
  virtual OutputTask *copy() {
    return new OutputTask(matrix, this->fullMatrixWidth, this->fullMatrixHeight, this->blockSize);
  }

 private:

  double *matrix;
  long fullMatrixWidth;
  long fullMatrixHeight;
  int blockSize;
  int numBlocksRows;
  int numBlocksCols;

};
#endif //HTGS_OUTPUTTASK_H
