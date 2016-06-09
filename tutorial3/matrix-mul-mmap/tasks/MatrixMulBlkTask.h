//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXMULBLKTASK_H
#define HTGS_MATRIXMULBLKTASK_H

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"
#include <htgs/api/ITask.hpp>

class MatrixMulBlkTask : public htgs::ITask<MatrixBlockMulData<double *>, MatrixBlockData<double *>>
{

 public:
  MatrixMulBlkTask(int numThreads, int fullMatrixWidthA, int fullMatrixHeightA, int fullMatrixWidthB, int fullMatrixHeightB, int blockSize) :
      ITask(numThreads), fullMatrixWidthA(fullMatrixWidthA), fullMatrixHeightA(fullMatrixHeightA),
      fullMatrixWidthB(fullMatrixWidthB), fullMatrixHeightB(fullMatrixHeightB), blockSize(blockSize)
  {}

  virtual ~MatrixMulBlkTask() {

  }
  virtual void initialize(int pipelineId,
                          int numPipeline) {

  }
  virtual void shutdown() {
  }

    virtual void executeTask(std::shared_ptr<MatrixBlockMulData<double *>> data) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    double * matrixA = matAData->getMatrixData();
    double * matrixB = matBData->getMatrixData();

    int width = matBData->getMatrixWidth();
    int height = matAData->getMatrixHeight();

      int blkRowA = data->getMatrixA()->getRequest()->getRow();
      int blkColA = data->getMatrixA()->getRequest()->getCol();
      int blkRowB = data->getMatrixB()->getRequest()->getRow();
      int blkColB = data->getMatrixB()->getRequest()->getCol();

    double *result = new double[width*height];

    for (int aRow = 0; aRow < height; aRow++)
    {
      for (int bCol = 0; bCol < width; bCol++)
      {
        double sum = 0.0;
        for (int k = 0; k < matAData->getMatrixWidth(); k++)
        {
          sum += matrixA[aRow*fullMatrixWidthA+k] *
              matrixB[k*fullMatrixWidthB+bCol];
        }
        result[aRow * width + bCol] = sum;
      }
    }

    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matAData->getRequest()->getRow(), matBData->getRequest()->getCol(), MatrixType::MatrixC));
      std::cout << "Computing A(" << matAData->getRequest()->getRow() << ", " << matAData->getRequest()->getCol() <<
          ") x B(" << matBData->getRequest()->getRow() << ", " << matBData->getRequest()->getCol() <<
          ") = C(" << matReq->getRow() << ", "<< matReq->getCol() << ")" <<std::endl;

    addResult(new MatrixBlockData<double *>(matReq, result, width, height));


  }
  virtual std::string getName() {
    return "MatrixMulBlkTask";
  }
  virtual MatrixMulBlkTask *copy() {
    return new MatrixMulBlkTask(this->getNumThreads(), fullMatrixWidthA, fullMatrixHeightA, fullMatrixWidthB, fullMatrixHeightB, blockSize);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }

 private:
  int fullMatrixWidthA;
  int fullMatrixHeightA;
  int fullMatrixWidthB;
  int fullMatrixHeightB;
  int blockSize;
};

#endif //HTGS_MATRIXMULBLKTASK_H
