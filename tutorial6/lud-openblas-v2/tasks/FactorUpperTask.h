//
// Created by tjb3 on 7/28/16.
//

#ifndef HTGS_TUTORIALS_FACTORUPPERTASK_H
#define HTGS_TUTORIALS_FACTORUPPERTASK_H

#include <htgs/api/ITask.hpp>
#include <cblas.h>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixFactorData.h"
#include "../data/MatrixPanelData.h"
class FactorUpperTask : public htgs::ITask<MatrixFactorData<double *>, MatrixPanelData>
{
 public:

  FactorUpperTask(int numThreads, long fullMatrixHeight, long fullMatrixWidth, long blockSize, int numBlocksHeight) :
      ITask(numThreads), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth), blockSize(blockSize), numBlocksHeight(numBlocksHeight) {}


  virtual void executeTask(std::shared_ptr<MatrixFactorData<double *>> data) {

    double *matrix = data->getUnFactoredMatrix()->getMatrixData();
    double *triMatrix = data->getTriangleMatrix()->getMatrixData();

    int height = (int) data->getTriangleMatrix()->getMatrixHeight();
    int width = (int) data->getTriangleMatrix()->getMatrixWidth();

    // Perform LZ = B
    // Overwrites matrix B
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, height, width, 1.0, triMatrix, fullMatrixHeight, matrix, fullMatrixHeight);

    int panelCol = data->getUnFactoredMatrix()->getRequest()->getCol();
    int panelRow = data->getUnFactoredMatrix()->getRequest()->getRow();

    int activeDiagonal = data->getTriangleMatrix()->getRequest()->getCol();

    long panelHeight = blockSize * (numBlocksHeight-panelRow);

    // Unfactored is now factored
    MatrixPanelData *panel = new MatrixPanelData(panelHeight, blockSize, panelCol, activeDiagonal, PanelState::TOP_FACTORED);
    panel->setMemory(matrix);

    addResult(panel);

  }
  virtual FactorUpperTask *copy() {
    return new FactorUpperTask(this->getNumThreads(), fullMatrixHeight, fullMatrixWidth, blockSize, numBlocksHeight);
  }

  virtual std::string getName() {
    return "FactorUpperTask";
  }

 private:
  long fullMatrixWidth;
  long fullMatrixHeight;
  long blockSize;
  int numBlocksHeight;
};

#endif //HTGS_TUTORIALS_FACTORUPPERTASK_H
