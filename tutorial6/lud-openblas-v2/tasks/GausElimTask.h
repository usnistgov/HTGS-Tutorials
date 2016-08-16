//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMTASK_H
#define HTGS_TUTORIALS_GAUSELIMTASK_H

#include <htgs/api/ITask.hpp>
#include <cblas.h>
#include <lapacke.h>
#include "../data/MatrixBlockData.h"
#include "../../../tutorial-utils/util-matrix.h"
class GausElimTask : public htgs::ITask<MatrixBlockData<double *>, MatrixBlockData<double *>>
{
 public:
  GausElimTask(int numThreads, long fullMatrixHeight, long fullMatrixWidth, int blockSize) :
  ITask(numThreads), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth), blockSize(blockSize)
  {
    this->ipiv = new long long int[blockSize];
  }

  ~GausElimTask()
  {
    delete []ipiv;
  }

  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {

    double *matrix = data->getMatrixData();

    LAPACKE_dgetf2(LAPACK_COL_MAJOR, data->getMatrixHeight(), data->getMatrixWidth(), matrix, fullMatrixHeight, (int *)ipiv);

    addResult(data);
  }
  virtual GausElimTask *copy() {
    return new GausElimTask(this->getNumThreads(), fullMatrixHeight, fullMatrixWidth, blockSize);
  }

  virtual std::string getName() {
    return "GausElimTask";
  }

 private:
  long fullMatrixWidth;
  long fullMatrixHeight;
  long long int *ipiv;
  int blockSize;
};

#endif //HTGS_TUTORIALS_GAUSELIMTASK_H
