//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXMULBLKTASK_H
#define HTGS_MATRIXMULBLKTASK_H

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"
#include <htgs/api/ITask.hpp>

class MatrixMulBlkTask : public htgs::ITask<MatrixBlockMulData<MatrixMemoryData_t>, MatrixBlockData<double *>>
{

 public:
  MatrixMulBlkTask(int numThreads) : ITask(numThreads) {}

  virtual ~MatrixMulBlkTask() {

  }
  virtual void initialize(int pipelineId,
                          int numPipeline) {

  }
  virtual void shutdown() {
  }

    virtual void executeTask(std::shared_ptr<MatrixBlockMulData<MatrixMemoryData_t>> data) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    MatrixMemoryData_t matrixA = matAData->getMatrixData();
    MatrixMemoryData_t matrixB = matBData->getMatrixData();

    int width = matBData->getMatrixWidth();
    int height = matAData->getMatrixHeight();

    double *result = new double[width*height];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, height, width, matAData->getMatrixWidth(), 1.0, matrixA->get(), matAData->getMatrixWidth(),
                  matrixB->get(), width, 0.0, result, width);

    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matAData->getRequest()->getRow(), matBData->getRequest()->getCol(), MatrixType::MatrixC));

//    std::cout << "Computing A(" << matAData->getRequest()->getRow() << ", " << matAData->getRequest()->getCol() <<
//          ") x B(" << matBData->getRequest()->getRow() << ", " << matBData->getRequest()->getCol() <<
//          ") = C(" << matReq->getRow() << ", "<< matReq->getCol() << ")" <<std::endl;

    addResult(new MatrixBlockData<double *>(matReq, result, width, height));

    this->memRelease("MatrixA", matrixA);
    this->memRelease("MatrixB", matrixB);

  }
  virtual std::string getName() {
    return "MatrixMulBlkTask";
  }
  virtual MatrixMulBlkTask *copy() {
    return new MatrixMulBlkTask(this->getNumThreads());
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }
};

#endif //HTGS_MATRIXMULBLKTASK_H
