//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_FACTORLOWERTASK_H
#define HTGS_TUTORIALS_FACTORLOWERTASK_H

#include <htgs/api/ICudaTask.hpp>
#include <cblas.h>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixFactorData.h"
class FactorLowerTask : public htgs::ICudaTask<MatrixFactorCudaData<double *>, MatrixBlockMultiData<double *>>
{
 public:

  FactorLowerTask(CUcontext *contexts, int *devIds, int numGPUs, long fullMatrixHeight, long fullMatrixWidth) : ICudaTask(contexts, devIds, numGPUs), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth) {}

  virtual void executeGPUTask(std::shared_ptr<MatrixFactorCudaData<double *>> data, CUstream stream) {

  }

  // TODO: Remove
  virtual void executeTask(std::shared_ptr<MatrixFactorCudaData<double *>> data, CUstream stream) {

//    double *matrix = data->getUnFactoredMatrix()->getMatrixData();
//    double *triMatrix = data->getTriangleMatrix()->getMatrixData();
//
//    int height = (int) data->getTriangleMatrix()->getMatrixHeight();
//    int width = (int) data->getTriangleMatrix()->getMatrixWidth();
//
//    // Perform WU = B
//    // Overwrites matrix B
//    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, height, width, 1.0, triMatrix, fullMatrixHeight, matrix, fullMatrixHeight);

    // TODO: Unfactored is now factored
//    addResult(data->getUnFactoredMatrix());

  }
  virtual FactorLowerTask *copy() {
    return new FactorLowerTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs(), fullMatrixHeight, fullMatrixWidth);
  }



  virtual std::string getName() {
    return "FactorLowerTask";
  }

 private:
  long fullMatrixWidth;
  long fullMatrixHeight;
};

#endif //HTGS_TUTORIALS_FACTORLOWERTASK_H
