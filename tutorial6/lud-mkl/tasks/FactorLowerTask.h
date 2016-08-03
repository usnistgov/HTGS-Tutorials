//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_FACTORLOWERTASK_H
#define HTGS_TUTORIALS_FACTORLOWERTASK_H

#include <htgs/api/ITask.hpp>
#include <mkl.h>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixFactorData.h"
class FactorLowerTask : public htgs::ITask<MatrixFactorData<double *>, MatrixBlockData<double *>>
{
 public:

  FactorLowerTask(int numThreads, long fullMatrixHeight, long fullMatrixWidth) : ITask(numThreads), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth) {}


  virtual void executeTask(std::shared_ptr<MatrixFactorData<double *>> data) {

    double *matrix = data->getUnFactoredMatrix()->getMatrixData();
    double *triMatrix = data->getTriangleMatrix()->getMatrixData();

    int height = (int) data->getTriangleMatrix()->getMatrixHeight();
    int width = (int) data->getTriangleMatrix()->getMatrixWidth();

    // Perform WU = B
    // Overwrites matrix B
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, height, width, 1.0, triMatrix, fullMatrixHeight, matrix, fullMatrixHeight);

    // Unfactored is now factored
    addResult(data->getUnFactoredMatrix());

  }
  virtual FactorLowerTask *copy() {
    return new FactorLowerTask(this->getNumThreads(), fullMatrixHeight, fullMatrixWidth);
  }

  virtual std::string getName() {
    return "FactorLowerTask";
  }

 private:
  long fullMatrixWidth;
  long fullMatrixHeight;
};

#endif //HTGS_TUTORIALS_FACTORLOWERTASK_H
