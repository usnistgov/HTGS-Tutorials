//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_FACTORUPPERTASK_H
#define HTGS_TUTORIALS_FACTORUPPERTASK_H

#include <htgs/api/ITask.hpp>
#include <cblas.h>
#include "../../common/data/MatrixBlockData.h"
#include "../../common/data/MatrixFactorData.h"
#include "../../common/data/MatrixBlockMultiData.h"
class FactorUpperTask : public htgs::ITask<MatrixFactorData, MatrixBlockMultiData>
{
 public:

  FactorUpperTask(int numThreads, long fullMatrixHeight, long fullMatrixWidth) : ITask(numThreads), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth) {}


  virtual void executeTask(std::shared_ptr<MatrixFactorData> data) {

    double *matrix = data->getUnFactoredMatrix()->getMatrixData();
    double *triMatrix = data->getTriangleMatrix()->getMatrixData();

    int height = (int) data->getTriangleMatrix()->getMatrixHeight();
    int width = (int) data->getTriangleMatrix()->getMatrixWidth();

    // Perform LZ = B
    // Overwrites matrix B
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, height, width, 1.0, triMatrix, fullMatrixHeight, matrix, fullMatrixHeight);

    // Unfactored is now factored
    auto multiData = new MatrixBlockMultiData(data->getUnFactoredMatrix(), nullptr);
    addResult(multiData);

  }
  virtual FactorUpperTask *copy() {
    return new FactorUpperTask(this->getNumThreads(), fullMatrixHeight, fullMatrixWidth);
  }

  virtual std::string getName() {
    return "FactorUpperTask";
  }

 private:
  long fullMatrixWidth;
  long fullMatrixHeight;
};

#endif //HTGS_TUTORIALS_FACTORUPPERTASK_H
