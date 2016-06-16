//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXACCUMTASK_H
#define HTGS_MATRIXACCUMTASK_H

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"
#include <htgs/api/ITask.hpp>

class MatrixAccumTask : public htgs::ITask<MatrixBlockMulData<double *>, MatrixBlockData<double *>>
{

 public:
  MatrixAccumTask(int numThreads) : ITask(numThreads) {}

  virtual ~MatrixAccumTask() {

  }
  virtual void initialize(int pipelineId,
                          int numPipeline) {

  }
  virtual void shutdown() {

  }
  virtual void executeTask(std::shared_ptr<MatrixBlockMulData<double *>> data) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    double *matrixA = matAData->getMatrixData();
    double *matrixB = matBData->getMatrixData();

    int width = matAData->getMatrixWidth();
    int height = matAData->getMatrixHeight();

    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        matrixA[i*width+j] += matrixB[i*width+j];
      }
    }

    delete [] matrixB;

//    auto matRequest = matAData->getRequest();
//    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matRequest->getRow(), matRequest->getCol(), MatrixType::MatrixC));

    addResult(data->getMatrixA());

  }
  virtual std::string getName() {
    return "MatrixAccumTask";
  }
  virtual MatrixAccumTask *copy() {
    return new MatrixAccumTask(this->getNumThreads());
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }
};

#endif //HTGS_MATRIXACCUMTASK_H
